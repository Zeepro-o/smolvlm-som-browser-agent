import os
import re
import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----------------------------------------------------------------------------
# 0. Reproducibility
# ----------------------------------------------------------------------------
SEED = 42
set_seed(SEED)

# ----------------------------------------------------------------------------
# 1. Configuration & Paths
# ----------------------------------------------------------------------------
MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
DATASET_PATH = "dataset_som/train_dataset.jsonl"
OUTPUT_DIR = "./som_smolvlm_lora_adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 2. 4-Bit Quantization Config (QLoRA)
#    FIX: skip the vision tower / connector / lm_head from 4-bit quantization.
#    Quantizing the vision encoder tends to hurt visual grounding quality far
#    more than it saves VRAM (it's already small relative to the LLM).
#    If you hit OOM on your 8GB card, you can drop "vision" from this list
#    to quantize it too, at some cost to image understanding quality.
# ----------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["vision", "connector", "projector", "lm_head"],
)

# ----------------------------------------------------------------------------
# 3. Load Processor & Model
# ----------------------------------------------------------------------------
print("⏳ Loading processor and base model in 4-bit...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# FIX: ensure a pad token exists (some causal LM tokenizers have none by default)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

# FIX: right-padding is what you want for training with this label-masking scheme
# (left-padding is the generation-time convention, not the training one)
processor.tokenizer.padding_side = "right"

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# FIX: required alongside gradient checkpointing, otherwise you'll get
# incorrect/no gradients or a runtime warning about incompatible caching
model.config.use_cache = False

# ----------------------------------------------------------------------------
# 4. Prepare Model for LoRA
# ----------------------------------------------------------------------------
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# FIX: target only the language-model attention projections, not the vision
# encoder's (SigLIP attention layers are also named q_proj/k_proj/v_proj/o_proj,
# so the original plain string list was silently also adapting vision weights).
# The negative lookahead excludes any module whose full dotted path contains
# "vision", regardless of what the vision submodule happens to be called.
TARGET_MODULES_REGEX = r"^(?!.*vision).*(?:q_proj|k_proj|v_proj|o_proj)$"

matched_modules = [
    name for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear) and re.fullmatch(TARGET_MODULES_REGEX, name)
]
if not matched_modules:
    raise ValueError(
        "No target modules matched the LoRA regex. Run `print(model)` to inspect "
        "the actual module names in this checkpoint and adjust TARGET_MODULES_REGEX."
    )
print(f"LoRA will target {len(matched_modules)} linear layers, e.g. {matched_modules[:4]}")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=TARGET_MODULES_REGEX,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ----------------------------------------------------------------------------
# 5. Load Dataset
# ----------------------------------------------------------------------------
print("⏳ Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

required_columns = {"image", "messages"}
missing = required_columns - set(dataset.column_names)
if missing:
    raise ValueError(f"Dataset is missing required column(s): {missing}")


def process_example(item):
    """
    Turn one {"image": path, "messages": [...]} record into model inputs,
    with labels masked so loss is only computed on the assistant's response
    (not the system/user prompt or the image tokens).
    """
    img_path = item["image"]
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    image = Image.open(img_path).convert("RGB")

    messages = item["messages"]
    if messages[-1]["role"] != "assistant":
        raise ValueError(
            "Expected the last message in each example to be the assistant's "
            "response so it can be used as the training target."
        )

    # Full conversation, exactly as the model should learn to produce it.
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # Everything up to (and including) the "it's your turn, assistant" cue,
    # but WITHOUT the assistant's actual answer. This is a token-for-token
    # prefix of full_text for standard chat templates.
    prompt_text = processor.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    full_inputs = processor(text=[full_text], images=[[image]], return_tensors="pt")
    prompt_inputs = processor(text=[prompt_text], images=[[image]], return_tensors="pt")

    input_ids = full_inputs["input_ids"][0]
    attention_mask = full_inputs["attention_mask"][0]
    pixel_values = full_inputs["pixel_values"][0]

    prompt_len = prompt_inputs["input_ids"].shape[1]

    labels = input_ids.clone()
    labels[:prompt_len] = -100  # mask everything except the assistant's answer

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }


def collate_fn(batch):
    examples = [process_example(item) for item in batch]

    max_len = max(ex["input_ids"].shape[0] for ex in examples)
    pad_id = processor.tokenizer.pad_token_id

    input_ids, attention_mask, labels, pixel_values = [], [], [], []
    for ex in examples:
        pad_len = max_len - ex["input_ids"].shape[0]
        input_ids.append(
            torch.cat([ex["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([ex["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels.append(
            torch.cat([ex["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        # Assumes one image per example, all resized to the same shape by the
        # processor (true for SmolVLM's default fixed-size preprocessing).
        pixel_values.append(ex["pixel_values"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "pixel_values": torch.stack(pixel_values),
    }


# ----------------------------------------------------------------------------
# 6. Training Arguments — tuned for an 8GB VRAM card
# ----------------------------------------------------------------------------
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS = 2

# FIX: some transformers builds (notably recent dev/main installs) don't
# expose `warmup_ratio` on TrainingArguments. `warmup_steps` is the older,
# universally-supported equivalent, so compute the same ~5% warmup manually.
effective_batch_size = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS
steps_per_epoch = max(1, len(dataset) // effective_batch_size)
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = max(1, int(0.05 * total_steps))
print(f"Total optimization steps: {total_steps} | warmup steps: {warmup_steps}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=2e-4,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,  # FIX: avoid filling the disk with per-epoch checkpoints
    bf16=True,
    optim="paged_adamw_8bit",
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    report_to="none",
    seed=SEED,
    gradient_checkpointing=True,  # FIX: essential for fitting an 8GB card
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# ----------------------------------------------------------------------------
# 7. Trainer Initialization
# ----------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

print("\n🚀 Starting QLoRA Training on RTX 5050...")
trainer.train()

# ----------------------------------------------------------------------------
# 8. Save Final Adapter
# ----------------------------------------------------------------------------
print(f"\n💾 Saving fine-tuned LoRA adapter to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("✅ Training complete!")

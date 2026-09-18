import os
import re
import argparse
import torch
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from som_common import MODEL_ID, get_bnb_config

# ----------------------------------------------------------------------------
# 0. CLI args -- parameterized so a retrain can never again silently overwrite
#    a previous adapter (run 2 overwrote run 1's weights this way once already)
#    and so epoch checkpoints land somewhere eval_offline.py/test_agent.py can
#    be pointed at individually via --adapter-path.
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="QLoRA fine-tune for the SoM grounding adapter")
parser.add_argument("--dataset", type=str, default="dataset_som/train_dataset.jsonl")
parser.add_argument("--output-dir", type=str, required=True,
                     help="e.g. ./som_smolvlm_lora_adapter_v3 -- always use a NEW dir per retrain")
parser.add_argument("--epochs", type=int, default=3,
                     help="Bumped default 2->3: with per-epoch checkpoints + eval_offline.py --adapter-path, "
                          "there's no downside to training a bit longer and empirically checking which "
                          "epoch's held-out accuracy is actually best, instead of guessing from training "
                          "loss alone (which reflects fit to seen data, not generalization).")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ----------------------------------------------------------------------------
# 0b. Reproducibility
# ----------------------------------------------------------------------------
set_seed(args.seed)

DATASET_PATH = args.dataset
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. 4-Bit Quantization Config (QLoRA)
#    Shared with test_agent.py / eval_offline.py via som_common.get_bnb_config()
#    -- this is what prevents the train/inference config drift that already
#    caused one silent bug in this project (see som_common.py docstring).
# ----------------------------------------------------------------------------
bnb_config = get_bnb_config()

# ----------------------------------------------------------------------------
# 2. Load Processor & Model
# ----------------------------------------------------------------------------
print("⏳ Loading processor and base model in 4-bit...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

processor.tokenizer.padding_side = "right"

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.config.use_cache = False

# ----------------------------------------------------------------------------
# 3. Prepare Model for LoRA
# ----------------------------------------------------------------------------
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Target LM attention AND MLP/feed-forward projections, not just attention.
# The original QLoRA paper's own ablations found adapting all linear layers
# meaningfully outperforms attention-only at a still-small parameter cost --
# attention-only was leaving representational capacity on the table for
# mapping visual tokens to discrete element-ID tokens. Vision encoder excluded
# via the same negative-lookahead as before (SigLIP names attention layers
# q_proj/k_proj/v_proj/o_proj too, but its MLP layers are fc1/fc2, so this
# doesn't accidentally pull in vision weights -- verified via the printed
# module list below; check it after any transformers/model version change).
TARGET_MODULES_REGEX = r"^(?!.*vision).*(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

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
# 4. Load Dataset
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

    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
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
    labels[:prompt_len] = -100

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
        pixel_values.append(ex["pixel_values"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "pixel_values": torch.stack(pixel_values),
    }


# ----------------------------------------------------------------------------
# 5. Training Arguments -- tuned for an 8GB VRAM card
# ----------------------------------------------------------------------------
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS = args.epochs

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
    save_total_limit=NUM_EPOCHS,  # keep every epoch's checkpoint (not just last 2) for per-epoch eval
    bf16=True,
    optim="paged_adamw_8bit",
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    report_to="none",
    seed=args.seed,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# ----------------------------------------------------------------------------
# 6. Trainer Initialization
# ----------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

print(f"\n🚀 Starting QLoRA Training -> {OUTPUT_DIR} ({NUM_EPOCHS} epochs)...")
trainer.train()

# ----------------------------------------------------------------------------
# 7. Save Final Adapter
# ----------------------------------------------------------------------------
print(f"\n💾 Saving fine-tuned LoRA adapter to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("✅ Training complete!")
print(f"\nPer-epoch checkpoints are under {OUTPUT_DIR}/checkpoint-*/")
print("Compare them with: python eval_offline.py --data <holdout>.jsonl --adapter-path <checkpoint dir>")

import torch
from transformers import AutoModelForImageTextToText
from som_common import MODEL_ID, get_bnb_config

bnb_config = get_bnb_config()
print(f"Loading model with bnb_config:\n{bnb_config}\n")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map="auto"
)

print("\nChecking actual module types for vision / connector / projector / lm_head:\n")
checked = 0
any_quantized = False
for name, module in model.named_modules():
    cls_name = type(module).__name__
    if cls_name not in ("Linear4bit", "Linear"):
        continue
    if any(key in name for key in ["vision", "connector", "projector", "lm_head"]):
        quantized = cls_name == "Linear4bit"
        any_quantized = any_quantized or quantized
        dtype = getattr(module.weight, "dtype", "?")
        print(f"  {name:<60} class={cls_name:<12} quantized={quantized}  weight_dtype={dtype}")
        checked += 1

print()
if checked == 0:
    print("No matching modules found.")
elif any_quantized:
    print("=> WARNING: Target modules are STILL quantized to 4-bit.")
else:
    print("=> SUCCESS: All target modules are plain Linear (not quantized) in bfloat16!")

"""
Offline, quantitative evaluation of the fine-tuned SoM adapter.

Unlike test_agent.py (which re-scrapes live pages and requires eyeballing
the result), this scores the model against ground truth that's *already in*
a held-out dataset -- the same auto-labeled (image, instruction, target)
triples your capture pipeline produces for training, just for sites that
were never trained on. No live browser needed, fully reproducible, directly
comparable across training runs.

Workflow:
    1. Capture a small holdout set from sites NOT in your training sites.json:
         python capture_som_dataset.py --sites holdout_sites.json --output-dir dataset_som_holdout

    2. Score the adapter against it:
         python eval_offline.py --data dataset_som_holdout/train_dataset.jsonl

Metrics reported, overall and broken down by site:
    - json_valid_rate    : did the model even produce parseable JSON
    - action_match_rate  : predicted action type (click/type) == ground truth
    - element_id_match   : predicted element_id == ground truth element_id
                            (the main "did it ground correctly" number)
    - coord_error_px     : pixel distance between predicted and ground-truth
                            coordinate, median/mean. NOTE: as of the
                            coordinate-free retrain (see strip_coordinates.py),
                            the model is no longer trained to output a
                            "coordinate" field at all -- this is intentional,
                            not a bug. Coordinates are resolved at inference
                            time by looking up the chosen element_id's known
                            bbox from the same overlay-generation step that
                            produced the marks, not from the model's output.
                            This metric will show as "n/a (by design)" for
                            adapters trained after that change; it's kept
                            around so older adapters remain scoreable too.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
ADAPTER_PATH = "./som_smolvlm_lora_adapter"


def load_model():
    print("⏳ Loading processor and base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )  # same config as training -- see test_agent.py for why this matters

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, processor


def predict(model, processor, image: Image.Image, instruction: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[[image]], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=64, do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def parse_json_loose(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def load_records(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def score_one(model, processor, record: dict) -> dict:
    instruction = record["messages"][0]["content"][1]["text"]
    ground_truth = json.loads(record["messages"][1]["content"][0]["text"])

    image = Image.open(record["image"]).convert("RGB")
    raw_output = predict(model, processor, image, instruction)
    predicted = parse_json_loose(raw_output)

    result = {
        "site": record.get("site", "unknown"),
        "instruction": instruction,
        "raw_output": raw_output,
        "ground_truth": ground_truth,
        "predicted": predicted,
        "json_valid": predicted is not None,
        "action_match": False,
        "element_id_match": False,
        "coord_error_px": None,
    }

    if predicted is not None:
        result["action_match"] = predicted.get("action") == ground_truth.get("action")
        result["element_id_match"] = predicted.get("element_id") == ground_truth.get("element_id")

        pc, gc = predicted.get("coordinate"), ground_truth.get("coordinate")
        if isinstance(pc, dict) and isinstance(gc, dict) and "x" in pc and "y" in gc:
            try:
                result["coord_error_px"] = ((pc["x"] - gc["x"]) ** 2 + (pc["y"] - gc["y"]) ** 2) ** 0.5
            except (TypeError, KeyError):
                pass

    return result


def summarize(results: list) -> dict:
    n = len(results)
    coord_errors = [r["coord_error_px"] for r in results if r["coord_error_px"] is not None]

    summary = {
        "n": n,
        "json_valid_rate": sum(r["json_valid"] for r in results) / n,
        "action_match_rate": sum(r["action_match"] for r in results) / n,
        "element_id_match_rate": sum(r["element_id_match"] for r in results) / n,
        "coord_error_px_median": statistics.median(coord_errors) if coord_errors else None,
        "coord_error_px_mean": statistics.mean(coord_errors) if coord_errors else None,
    }

    by_site = defaultdict(list)
    for r in results:
        by_site[r["site"]].append(r)
    summary["by_site"] = {
        site: {
            "n": len(rs),
            "json_valid_rate": sum(r["json_valid"] for r in rs) / len(rs),
            "element_id_match_rate": sum(r["element_id_match"] for r in rs) / len(rs),
            "action_match_rate": sum(r["action_match"] for r in rs) / len(rs),
        }
        for site, rs in sorted(by_site.items())
    }
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Score the fine-tuned adapter against a held-out labeled dataset")
    parser.add_argument("--data", type=Path, required=True, help="Path to a held-out train_dataset.jsonl-format file")
    parser.add_argument("--out", type=Path, default=Path("eval_offline_report.json"))
    args = parser.parse_args()

    records = load_records(args.data)
    if not records:
        print(f"No records found in {args.data}")
        return

    sites_in_eval = {r.get("site", "unknown") for r in records}
    print(f"Loaded {len(records)} held-out records from {len(sites_in_eval)} sites: {sorted(sites_in_eval)}")

    model, processor = load_model()

    results = []
    for i, record in enumerate(records):
        r = score_one(model, processor, record)
        results.append(r)
        mark = "✅" if r["element_id_match"] else "❌"
        print(f"[{i + 1}/{len(records)}] {mark} {r['site']:<20} '{r['instruction'][:40]}' "
              f"-> pred={r['predicted']} gt={r['ground_truth']}")

    summary = summarize(results)

    print("\n" + "=" * 60)
    print(f"n = {summary['n']}")
    print(f"  JSON valid:       {summary['json_valid_rate']:.1%}")
    print(f"  Action match:     {summary['action_match_rate']:.1%}")
    print(f"  Element ID match: {summary['element_id_match_rate']:.1%}   <-- main grounding accuracy number")
    if summary["coord_error_px_median"] is not None:
        print(f"  Coord error (px): median {summary['coord_error_px_median']:.0f}, mean {summary['coord_error_px_mean']:.0f}")
    else:
        print("  Coord error (px): n/a (by design -- model no longer predicts coordinates,")
        print("                    they're resolved from the known element bbox at inference time)")
    print("\n  By site:")
    for site, s in summary["by_site"].items():
        print(f"    {site:<20} n={s['n']:<4} json_valid={s['json_valid_rate']:.1%}  "
              f"element_id_match={s['element_id_match_rate']:.1%}  action_match={s['action_match_rate']:.1%}")
    print("=" * 60)

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nFull report -> {args.out}")


if __name__ == "__main__":
    main()

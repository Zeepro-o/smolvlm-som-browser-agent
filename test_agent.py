"""
Inference + evaluation harness for the fine-tuned SoM grounding adapter.

Loads the base model + LoRA adapter ONCE, then runs it against a list of
(url, instruction) test cases and, for each one:
  - captures a fresh Set-of-Marks screenshot (same injection script used to
    build the training set -- occlusion check + bbox/center metadata, not
    just a raw count, so element numbering actually matches what the model
    was trained on)
  - asks the model to predict an action
  - resolves the predicted element_id back to a real tagged element
  - draws the predicted target on the screenshot so you can eyeball it
  - optionally (--execute) actually clicks/types it via Playwright and
    reports whether the page changed

IMPORTANT: the default test cases below are sites that are NOT in the
sites.json used to build the training set. `data.nasa.gov` (used in the
original draft of this script) IS one of the training sites (nasa_open_data),
so testing on it mostly measures memorization, not generalization -- swap in
your own held-out sites if you've since edited sites.json.

Usage:
    python test_agent.py                       # print-only, default cases
    python test_agent.py --execute              # also click the prediction
    python test_agent.py --cases my_cases.json  # your own [{url, instruction}, ...]
"""

from __future__ import annotations

import re
import json
import asyncio
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from playwright.async_api import async_playwright
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
ADAPTER_PATH = "./som_smolvlm_lora_adapter"
OUTPUT_DIR = Path("eval_results")

VIEWPORT = {"width": 1280, "height": 800}          # must match training capture
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SoMDatasetBot/1.0"
)  # matches the UA used to build the training set, to keep rendering consistent
NAV_TIMEOUT_MS = 25_000
EXTRA_WAIT_MS = 2_500
RETRY_WAIT_MS = 6_000

CONSENT_SELECTORS = [
    "button:has-text('Accept All')",
    "button:has-text('Accept all')",
    "button:has-text('Accept')",
    "button:has-text('I Agree')",
    "button:has-text('Allow all')",
    "button:has-text('Got it')",
    "#onetrust-accept-btn-handler",
    "[aria-label='Accept cookies']",
]

# Genuinely unseen sites (not in the 38-site sites.json starter) -- adjust
# the instruction wording once you see what's actually tagged on each page.
TEST_CASES = [
    {"url": "https://opendata.cern.ch/", "instruction": "Click on 'Search'"},
    {"url": "https://data.london.gov.uk/", "instruction": "Search using 'Search'"},
    {"url": "https://opendata.cityofnewyork.us/", "instruction": "Click on 'Data'"},
]

# Same injection script used by the capture pipeline: occlusion check +
# full per-element metadata (id/tag/text/bbox/center), not just a count.
SOM_INJECTION_SCRIPT = """
() => {
    const existing = document.querySelectorAll('.som-overlay-badge, .som-overlay-box');
    existing.forEach(el => el.remove());

    const interactiveSelectors = [
        'button', 'a', 'input', 'select', 'textarea',
        '[role="button"]', '[role="link"]', '[role="checkbox"]',
        '[role="menuitem"]', '[role="tab"]', '[onclick]'
    ];

    const elements = Array.from(document.querySelectorAll(interactiveSelectors.join(',')));
    const items = [];
    let count = 0;

    elements.forEach(el => {
        const rect = el.getBoundingClientRect();
        if (rect.width < 12 || rect.height < 12) return;
        if (rect.top < 0 || rect.top > window.innerHeight) return;
        if (rect.left < 0 || rect.left > window.innerWidth) return;

        const style = window.getComputedStyle(el);
        if (style.visibility === 'hidden' || style.display === 'none' || style.opacity === '0') return;

        const rawText = (el.innerText || el.getAttribute('placeholder') || el.getAttribute('aria-label') || el.getAttribute('title') || el.value || '').trim();
        if (!rawText || rawText.length < 2 || rawText.length > 60) return;

        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;
        const topEl = document.elementFromPoint(cx, cy);
        if (!topEl || (!el.contains(topEl) && !topEl.contains(el))) return;

        count += 1;

        const box = document.createElement('div');
        box.className = 'som-overlay-box';
        box.style.position = 'fixed';
        box.style.left = `${rect.left}px`;
        box.style.top = `${rect.top}px`;
        box.style.width = `${rect.width}px`;
        box.style.height = `${rect.height}px`;
        box.style.border = '2px solid #00FF66';
        box.style.backgroundColor = 'rgba(0, 255, 102, 0.08)';
        box.style.pointerEvents = 'none';
        box.style.zIndex = '999998';
        document.body.appendChild(box);

        const badge = document.createElement('div');
        badge.className = 'som-overlay-badge';
        badge.innerText = `[${count}]`;
        badge.style.position = 'fixed';
        badge.style.left = `${Math.max(0, rect.left)}px`;
        badge.style.top = `${Math.max(0, rect.top - 18)}px`;
        badge.style.backgroundColor = '#00FF66';
        badge.style.color = '#000000';
        badge.style.fontSize = '12px';
        badge.style.fontWeight = 'bold';
        badge.style.fontFamily = 'monospace';
        badge.style.padding = '1px 4px';
        badge.style.borderRadius = '3px';
        badge.style.boxShadow = '0 0 4px rgba(0,0,0,0.8)';
        badge.style.pointerEvents = 'none';
        badge.style.zIndex = '999999';
        document.body.appendChild(badge);

        items.push({
            id: count,
            tag: el.tagName.toLowerCase(),
            type: el.getAttribute('type') || null,
            text: rawText.slice(0, 80),
            role: el.getAttribute('role') || null,
            bbox: {
                x: Math.round(rect.left), y: Math.round(rect.top),
                width: Math.round(rect.width), height: Math.round(rect.height)
            },
            center: { x: Math.round(cx), y: Math.round(cy) }
        });
    });

    return items;
}
"""


# --------------------------------------------------------------------------
# Browser side: capture + optional action execution
# --------------------------------------------------------------------------

async def dismiss_consent_banners(page):
    for selector in CONSENT_SELECTORS:
        try:
            await page.locator(selector).first.click(timeout=1200)
            await page.wait_for_timeout(300)
            return
        except Exception:
            continue


async def load_and_tag(page, url: str, settle_ms: int) -> list:
    await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
    try:
        await page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        pass
    await dismiss_consent_banners(page)
    await page.wait_for_timeout(settle_ms)
    return await page.evaluate(SOM_INJECTION_SCRIPT)


async def capture_som(context, url: str, name: str):
    page = await context.new_page()
    page.set_default_timeout(NAV_TIMEOUT_MS)

    elements = await load_and_tag(page, url, EXTRA_WAIT_MS)
    if not elements:
        print("  0 elements on first pass — retrying with a longer settle time")
        elements = await load_and_tag(page, url, RETRY_WAIT_MS)

    screenshot_path = OUTPUT_DIR / f"{name}_som.png"
    await page.screenshot(path=str(screenshot_path))
    return screenshot_path, elements, page


async def execute_action_on_page(page, element: dict, action: dict) -> dict:
    before_url = page.url
    cx, cy = element["center"]["x"], element["center"]["y"]
    try:
        await page.mouse.click(cx, cy)
        if action.get("action") == "type":
            await page.keyboard.type(str(action.get("value", "")), delay=20)
        await page.wait_for_timeout(1500)
        return {"executed": True, "before_url": before_url, "after_url": page.url, "title": await page.title()}
    except Exception as e:
        return {"executed": False, "error": str(e), "before_url": before_url, "after_url": page.url}


# --------------------------------------------------------------------------
# Model side
# --------------------------------------------------------------------------

def load_model():
    print("⏳ Loading processor and base model in 4-bit...")
    # Must match train_som_qlora.py's BitsAndBytesConfig exactly. The adapter's
    # weights were optimized against gradients computed with THIS quantization
    # setup (vision/connector/lm_head excluded from 4-bit) -- evaluating with a
    # different setup introduces a train/inference mismatch and confounds
    # "is my model good". If you change this in train_som_qlora.py, change it
    # here and in eval_offline.py too, in the same commit.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["vision", "connector", "projector", "lm_head"],
    )

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
    # tokenize=False is required here: apply_chat_template defaults to
    # tokenize=True, which returns token IDs, not a string -- feeding those
    # into processor(text=...) (which expects text) is the bug in the
    # original draft of this script.
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[[image]], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def parse_action(raw_text: str) -> dict | None:
    """Model was trained to emit only JSON, but this is defensive in case of
    trailing garbage -- which is itself a useful signal: if you see garbage
    after a valid JSON blob, it likely means real end-of-sequence tokens got
    masked out of the loss during training (a padding/EOS bug), not that
    this parser failed."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def match_element(elements: list, element_id) -> dict | None:
    for el in elements:
        if el["id"] == element_id:
            return el
    return None


def annotate_prediction(screenshot_path: Path, element: dict | None, out_path: Path):
    img = Image.open(screenshot_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    if element is not None:
        b = element["bbox"]
        x0, y0, x1, y1 = b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]
        draw.rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline=(255, 0, 200), width=4)
        draw.text((x0, max(0, y0 - 34)), "PREDICTED", fill=(255, 0, 200))
    else:
        draw.rectangle([10, 10, 230, 40], fill=(255, 0, 200))
        draw.text((16, 16), "NO ELEMENT MATCH", fill=(255, 255, 255))
    img.save(out_path)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

async def run_all(test_cases: list, execute_action: bool):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, processor = load_model()

    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport=VIEWPORT, user_agent=USER_AGENT, locale="en-US")

        for i, case in enumerate(test_cases):
            name = f"case{i}_{re.sub(r'[^A-Za-z0-9]', '_', case['url'])[:40]}"
            print(f"\n--- [{i + 1}/{len(test_cases)}] {case['url']}  ::  {case['instruction']}")

            screenshot_path, elements, page = await capture_som(context, case["url"], name)
            print(f"  tagged {len(elements)} elements")

            image = Image.open(screenshot_path).convert("RGB")
            raw_output = predict(model, processor, image, case["instruction"])
            print(f"  model output: {raw_output}")

            parsed = parse_action(raw_output)
            matched = match_element(elements, parsed.get("element_id")) if parsed else None

            annotate_prediction(screenshot_path, matched, OUTPUT_DIR / f"{name}_predicted.png")

            if matched:
                print(f"  -> predicted element [{parsed['element_id']}]: '{matched['text']}'")
            else:
                print(f"  -> could not resolve element_id to a tagged element (parsed={parsed})")

            result = {
                "url": case["url"],
                "instruction": case["instruction"],
                "raw_output": raw_output,
                "parsed_action": parsed,
                "matched_element_text": matched["text"] if matched else None,
                "annotated_screenshot": str(OUTPUT_DIR / f"{name}_predicted.png"),
            }

            if execute_action and matched and parsed:
                exec_result = await execute_action_on_page(page, matched, parsed)
                result["execution"] = exec_result
                print(f"  -> executed: {exec_result}")

            await page.close()
            results.append(result)

        await browser.close()

    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} case results -> {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the fine-tuned SoM adapter against test sites")
    parser.add_argument("--execute", action="store_true", help="Actually click/type the predicted element via Playwright")
    parser.add_argument("--cases", type=Path, default=None, help="Optional JSON file: [{\"url\":..., \"instruction\":...}, ...]")
    args = parser.parse_args()

    test_cases = TEST_CASES
    if args.cases:
        with open(args.cases) as f:
            test_cases = json.load(f)

    asyncio.run(run_all(test_cases, execute_action=args.execute))


if __name__ == "__main__":
    main()

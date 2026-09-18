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
  - optionally (--execute) actually clicks/types it via Playwright, checks
    whether the URL or the visible element set changed afterward, and -- if
    nothing changed -- asks the model again for a different element, up to
    --max-attempts tries. This is a verified-retry loop, not a one-shot
    click: a single ~50% one-shot grounding accuracy compounds into a much
    higher effective task-completion rate across a few cheap retries, since
    "did the click do anything" is easy to check and most instructions have
    more than one plausible target if the first guess misses.

IMPORTANT: the default test cases below are sites that are NOT in the
sites.json used to build the training set. `data.nasa.gov` (used in the
original draft of this script) IS one of the training sites (nasa_open_data),
so testing on it mostly measures memorization, not generalization -- swap in
your own held-out sites if you've since edited sites.json.

Usage:
    python test_agent.py                       # print-only, default cases
    python test_agent.py --execute              # click + verify + retry on miss
    python test_agent.py --execute --max-attempts 5
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
from transformers import AutoProcessor, AutoModelForImageTextToText

from som_common import (
    MODEL_ID, DEFAULT_ADAPTER_PATH, VIEWPORT, USER_AGENT, NAV_TIMEOUT_MS, EXTRA_WAIT_MS,
    RETRY_WAIT_MS, SOM_INJECTION_SCRIPT, get_bnb_config, load_and_tag,
)

ADAPTER_PATH = DEFAULT_ADAPTER_PATH
OUTPUT_DIR = Path("eval_results")

# Genuinely unseen sites (not in the 38-site sites.json starter) -- adjust
# the instruction wording once you see what's actually tagged on each page.
TEST_CASES = [
    {"url": "https://opendata.cern.ch/", "instruction": "Click on 'Search'"},
    {"url": "https://data.london.gov.uk/", "instruction": "Search using 'Search'"},
    {"url": "https://opendata.cityofnewyork.us/", "instruction": "Click on 'Data'"},
]

# SOM_INJECTION_SCRIPT, dismiss_consent_banners, and load_and_tag now live in
# som_common.py (imported above), so a live inference capture here is
# guaranteed to use the exact same overlay marks the training data was built
# with -- badge styling changes only need to happen in one place.


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


def page_state_fingerprint(elements: list) -> frozenset:
    """A cheap signature of 'what's on the page right now', built from the
    same (tag, text) pairs dedupe_ambiguous() uses to spot duplicates. Two
    fingerprints being equal is a reasonable proxy for 'the click did not
    change anything visible' -- new elements appearing (a dropdown opened,
    a modal appeared, navigation happened) or existing ones disappearing
    will change the set. It won't catch every kind of change (e.g. a value
    updating inside an element that keeps the same tag+text), but it's a
    cheap, generalizable signal that needs no per-instruction expectations."""
    return frozenset((el["tag"], el["text"].strip().lower()) for el in elements)


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


async def get_input_value(page, element: dict) -> str | None:
    """Read the current value of a form input via its DOM center point --
    used to verify 'type' actions, since typing into a field rarely changes
    the visible interactive-element set the way a click/navigation does."""
    cx, cy = element["center"]["x"], element["center"]["y"]
    try:
        return await page.evaluate(
            "([cx, cy]) => { const el = document.elementFromPoint(cx, cy); "
            "return el ? (el.value ?? null) : null; }",
            [cx, cy],
        )
    except Exception:
        return None


async def execute_with_verification(
    page, image: Image.Image, instruction: str, elements: list,
    model, processor, max_attempts: int = 3,
) -> dict:
    """Click (or type into) the model's top prediction, then verify it
    actually did something before trusting it:
      - action == 'type': re-read the input's value and check it now
        contains what we typed. Typing rarely changes the visible element
        set, so a DOM diff is the wrong check for this action type.
      - action == 'click': if the URL changed, that's an immediate success
        -- skip the DOM diff entirely, it's redundant once we already know
        navigation happened. Otherwise fall back to comparing the visible
        element set before/after (catches in-page changes like a dropdown
        or modal opening that a URL check alone would miss).

    On a miss, re-prompt the model with plain negative feedback naming the
    element_id(s) that already failed and ask it to choose again.
    max_attempts is TOTAL tries including the first -- default 3 means the
    first attempt plus up to 2 retries, rather than anything logit-level.
    """
    before_state = page_state_fingerprint(elements)
    before_url = page.url

    tried_ids = []
    attempts = []
    current_instruction = instruction

    for attempt_num in range(1, max_attempts + 1):
        raw_output = predict(model, processor, image, current_instruction)
        parsed = parse_action(raw_output)
        matched = match_element(elements, parsed.get("element_id")) if parsed else None

        attempt_record = {
            "attempt": attempt_num,
            "raw_output": raw_output,
            "parsed_action": parsed,
            "matched_element_text": matched["text"] if matched else None,
        }

        if not matched or not parsed:
            attempt_record["outcome"] = "no_element_match"
            attempts.append(attempt_record)
            break  # can't click nothing -- no point retrying with the same unresolved output

        is_type_action = parsed.get("action") == "type"
        value_before = await get_input_value(page, matched) if is_type_action else None

        tried_ids.append(parsed["element_id"])
        exec_result = await execute_action_on_page(page, matched, parsed)
        attempt_record["execution"] = exec_result

        if not exec_result.get("executed"):
            attempt_record["outcome"] = "execution_error"
            attempts.append(attempt_record)
            break  # a Playwright-level error (e.g. detached element) won't fix itself on retry

        url_changed = exec_result["after_url"] != before_url
        attempt_record["url_changed"] = url_changed

        if url_changed:
            # Fast path: navigation happened, that's a success on its own --
            # no need to also diff the DOM.
            attempt_record["outcome"] = "success"
            attempts.append(attempt_record)
            return {"success": True, "attempts": attempts, "final_matched_element_text": matched["text"]}

        if is_type_action:
            value_after = await get_input_value(page, matched)
            typed_value = str(parsed.get("value", ""))
            value_changed = bool(value_after) and value_after != value_before and (
                typed_value == "" or typed_value.strip().lower() in value_after.strip().lower()
            )
            attempt_record["value_before"] = value_before
            attempt_record["value_after"] = value_after
            attempt_record["outcome"] = "success" if value_changed else "no_visible_change"
            attempts.append(attempt_record)
            if value_changed:
                return {"success": True, "attempts": attempts, "final_matched_element_text": matched["text"]}
        else:
            after_elements = await page.evaluate(SOM_INJECTION_SCRIPT)
            after_state = page_state_fingerprint(after_elements)
            dom_changed = after_state != before_state
            attempt_record["dom_changed"] = dom_changed
            attempt_record["outcome"] = "success" if dom_changed else "no_visible_change"
            attempts.append(attempt_record)
            if dom_changed:
                return {"success": True, "attempts": attempts, "final_matched_element_text": matched["text"]}

        # No visible change -- plain re-prompt with negative feedback naming
        # EVERY element_id tried so far (not just the most recent one), so
        # the model can't cycle back to an earlier failed guess on the next
        # attempt. No logit/token-level intervention, just prompt text.
        failed_list = ", ".join(f"[{i}]" for i in tried_ids)
        current_instruction = (
            f"{instruction}\n\n"
            f"(Clicking elements {failed_list} caused no page change. "
            f"Choose a different element.)"
        )

    return {
        "success": False,
        "attempts": attempts,
        "final_matched_element_text": None,
    }


# --------------------------------------------------------------------------
# Model side
# --------------------------------------------------------------------------

def load_model():
    print("⏳ Loading processor and base model in 4-bit...")
    bnb_config = get_bnb_config()  # shared with train_som_qlora.py -- see som_common.py

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
    """str() comparison, not ==: element_id is stored as an int (see
    som_common.py's count += 1), but nothing guarantees the model always
    emits element_id as a JSON integer rather than a string -- 4 == "4" is
    False in Python, which would silently fail to match a perfectly valid
    prediction. Comparing string forms sidesteps that without needing to
    trust or validate the type the model happened to emit."""
    for el in elements:
        if str(el["id"]) == str(element_id):
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

async def run_all(test_cases: list, execute_action: bool, max_attempts: int = 3):
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

            if execute_action:
                # Verified retry loop: click, check if anything visible
                # changed, retry with a different guess if not.
                outcome = await execute_with_verification(
                    page, image, case["instruction"], elements, model, processor,
                    max_attempts=max_attempts,
                )
                first_attempt = outcome["attempts"][0] if outcome["attempts"] else {}
                raw_output = first_attempt.get("raw_output", "")
                parsed = first_attempt.get("parsed_action")
                matched = match_element(elements, parsed.get("element_id")) if parsed else None

                print(f"  model output (attempt 1): {raw_output}")
                for a in outcome["attempts"]:
                    print(f"  attempt {a['attempt']}: {a.get('parsed_action')} -> {a['outcome']}")
                print(f"  -> verified success: {outcome['success']}")

                annotate_prediction(screenshot_path, matched, OUTPUT_DIR / f"{name}_predicted.png")

                result = {
                    "url": case["url"],
                    "instruction": case["instruction"],
                    "matched_element_text": matched["text"] if matched else None,
                    "annotated_screenshot": str(OUTPUT_DIR / f"{name}_predicted.png"),
                    "verified_success": outcome["success"],
                    "attempts": outcome["attempts"],
                }
            else:
                # Print-only path, unchanged: single prediction, no clicking.
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

            await page.close()
            results.append(result)

        await browser.close()

    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} case results -> {report_path}")

    if execute_action:
        verified = sum(1 for r in results if r.get("verified_success"))
        total_attempts = sum(len(r.get("attempts", [])) for r in results)
        print(f"\nVerified success: {verified}/{len(results)} cases "
              f"({verified / len(results):.1%})  |  total attempts used: {total_attempts}")


def main():
    parser = argparse.ArgumentParser(description="Run the fine-tuned SoM adapter against test sites")
    parser.add_argument("--execute", action="store_true", help="Actually click/type the predicted element via Playwright")
    parser.add_argument("--max-attempts", type=int, default=3,
                         help="With --execute, TOTAL tries per instruction including the first "
                              "(default 3 = 1 attempt + up to 2 retries) if a click/type produced "
                              "no verifiable change")
    parser.add_argument("--cases", type=Path, default=None, help="Optional JSON file: [{\"url\":..., \"instruction\":...}, ...]")
    parser.add_argument("--adapter-path", type=str, default=None, help="Override adapter dir, e.g. to test a specific epoch checkpoint")
    args = parser.parse_args()

    if args.adapter_path:
        global ADAPTER_PATH
        ADAPTER_PATH = args.adapter_path

    test_cases = TEST_CASES
    if args.cases:
        with open(args.cases) as f:
            test_cases = json.load(f)

    asyncio.run(run_all(test_cases, execute_action=args.execute, max_attempts=args.max_attempts))


if __name__ == "__main__":
    main()

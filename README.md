# On-Device Set-of-Mark Visual Grounding for a Lightweight Browser Agent

A small vision-language model (SmolVLM-Instruct, ~2.2B params) fine-tuned with QLoRA to look at
a screenshot with numbered "Set-of-Mark" (SoM) overlays on interactive elements, read a
natural-language instruction, and output *which numbered element* to act on -- the core
perception step for a lightweight browser-automation agent. Trains and runs on a single
8GB-VRAM consumer GPU (developed on an RTX 5050).

## How it works

1. **`sites.json`** -- list of target websites to scrape (`{name, url, category}`).
2. **`capture_som_dataset.py`** -- Playwright script. Visits each site, injects numbered
   overlay marks on interactive DOM elements (with an occlusion check so hidden/covered
   elements aren't tagged), screenshots the page, and auto-labels `(image, instruction,
   target JSON)` training records. Supports concurrency, per-host rate limiting, robots.txt
   compliance, and resume (skips sites already scraped).
3. **`train_som_qlora.py`** -- QLoRA fine-tuning. 4-bit NF4 quantization with the vision
   tower/connector/lm_head excluded from quantization, LoRA applied only to the language
   model's attention projections, gradient checkpointing, and labels masked so loss is only
   computed on the assistant's JSON response (not the prompt/instruction tokens).
4. **`eval_offline.py`** -- scores the adapter against a held-out labeled dataset, no live
   browser required. Reports `element_id_match_rate` (the main metric), `action_match_rate`,
   and `json_valid_rate`, broken down per site.
5. **`test_agent.py`** -- live sanity check: captures a fresh screenshot of a real site,
   runs the model, draws the predicted element on the screenshot, and (optionally) actually
   executes the click via Playwright.
6. **`holdout_sites.json`** -- sites deliberately never scraped into training data, used only
   for generalization eval.

## Key design decision: no coordinate prediction

The model was originally trained to output `{"action", "element_id", "coordinate": {"x","y"}}`.
Evaluation showed pixel-coordinate regression was essentially unlearnable at this data scale
(median error ~410px on a 1280x800 canvas -- close to random) and showed clear memorization
signatures (identical round-number coordinates reused across unrelated sites). It's also
unnecessary: the agent's own overlay-generation code already knows every `element_id`'s real
coordinate the moment it draws the marks, so the model only needs to name *which* element --
the agent's code looks up the actual click location, never trusting a model-predicted
coordinate. Dropping coordinates from the training target simplified the task and measurably
improved grounding accuracy (see Results).

## Results

Evaluated on 6 held-out sites never seen during training (`holdout_sites.json`, n=93):

| Run | Data | Coordinates in target | `element_id_match` |
|---|---|---|---|
| 1 | 32 sites, 481 records, 3 epochs | yes | 40.9% |
| 2 | 60 sites, 941 records, 2 epochs | no  | **50.5%** (improved on every site) |

Cross-site UI grounding on unseen sites is a known hard, unsolved problem even for large
frontier VLMs (see academic benchmarks like Mind2Web / WebArena for comparable or lower
ranges). A ~6.3M-parameter LoRA adapter on well under 1,000 examples was never going to hit
near-100% one-shot accuracy on arbitrary unseen sites -- the intended path to a reliable agent
is pairing this model with a verify-and-retry loop in the agent logic (click, check whether the
page changed as expected, retry if not), not chasing higher one-shot accuracy in isolation.

## Known limitations / next steps

- **Element-ID collapse bias**: on the weakest sites, wrong predictions cluster on a small set
  of low-numbered IDs regardless of instruction, while missed ground-truth IDs skew high
  (dense pages). Likely cause: training data underrepresents high-ID/dense-page elements
  relative to low-ID/common-nav elements. Rebalancing sampling is the highest-leverage
  remaining improvement.
- No verify-and-retry loop in the agent yet.
- A handful of source sites return 0 taggable elements even after a longer settle-time retry
  (likely bot detection / heavy client-side rendering) and don't contribute training data.

## Setup

```bash
python3 -m venv vlm-env
source vlm-env/bin/activate

# install torch first, matching your CUDA version:
# https://pytorch.org/get-started/locally/
pip install -r requirements.txt
playwright install chromium
```

## Usage

```bash
# 1. build the training set
python capture_som_dataset.py

# 2. train
python train_som_qlora.py

# 3. quantitative eval against held-out sites
python capture_som_dataset.py --sites holdout_sites.json --output-dir dataset_som_holdout
python eval_offline.py --data dataset_som_holdout/train_dataset.jsonl

# 4. live sanity check on a real page
python test_agent.py
```

## License

MIT License

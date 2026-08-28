"""
Set-of-Marks (SoM) dataset capture — scaled for 200+ target sites.

Loads target pages from a JSON config, tags visible/unoccluded interactive
elements with numbered overlay marks, screenshots the result, and emits a
JSONL training set pairing (image, instruction) -> structured action.

Why this looks different from a 2-site test script:
  - Records are flushed to disk per-site (records/{name}.jsonl), not held in
    memory until the end. A crash at site 150/200 doesn't lose sites 1-149.
  - Sites run concurrently (bounded globally and per-host), not one at a time.
  - A finished site is skipped on rerun (resume-by-default), so a second pass
    after a partial failure doesn't redo work or re-hit servers you already hit.
  - If a page yields 0 elements, it's retried once with a longer settle time
    before being given up on (slow client-side rendering is common across a
    diverse set of 200 sites; a fixed sleep that works for one site won't work
    for all of them).
  - Cookie/consent-banner dismissal and an occlusion check are back in — with
    only 2 sites you might not hit a blocking modal; across 200 you will.
  - A lightweight, fail-open robots.txt check runs before each site.
  - Prompts are sampled from a small template pool instead of one fixed
    phrasing, and each site is capped to a random sample of its elements so
    one JS-heavy site can't dominate the dataset.

Usage:
    python capture_som_dataset.py                     # sites.json, 8 concurrent
    python capture_som_dataset.py --concurrency 15
    python capture_som_dataset.py --force              # ignore resume cache
    python capture_som_dataset.py --merge-only         # just rebuild the merged JSONL
    python capture_som_dataset.py --sites my_sites.json --seed 42
"""

from __future__ import annotations

import re
import json
import random
import asyncio
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

OUTPUT_DIR = Path("dataset_som")
IMAGES_DIR = OUTPUT_DIR / "images"
RECORDS_DIR = OUTPUT_DIR / "records"          # one JSONL per site: crash-safe + resumable
FINAL_JSONL = OUTPUT_DIR / "train_dataset.jsonl"
SITES_CONFIG = Path(__file__).parent / "sites.json"

DEFAULT_CONCURRENCY = 8          # total pages in flight at once
PER_HOST_CONCURRENCY = 2         # max concurrent pages hitting the same domain
MAX_RECORDS_PER_SITE = 20        # cap so a handful of complex sites don't dominate the dataset
NAV_TIMEOUT_MS = 25_000
EXTRA_WAIT_MS = 2_500            # settle time after DOMContentLoaded, first attempt
RETRY_WAIT_MS = 6_000            # longer settle time used only on the zero-elements retry
REQUEST_JITTER_S = (0.2, 1.2)    # small randomized delay before each navigation

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SoMDatasetBot/1.0"
)

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

PROMPT_TEMPLATES_CLICK = [
    "Click on '{label}'",
    "Select '{label}'",
    "Press the '{label}' button",
    "Choose '{label}'",
    "Tap '{label}'",
]
PROMPT_TEMPLATES_TYPE = [
    "Enter a search query into '{label}'",
    "Type into '{label}'",
    "Fill in '{label}' with a search term",
    "Search using '{label}'",
]
TYPE_VALUES = [
    "Earth observation data",
    "climate change",
    "satellite imagery",
    "biodiversity dataset",
    "renewable energy statistics",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("som_capture")

# JavaScript snippet that injects Set-of-Marks overlays into the live page.
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

        // Occlusion check: is this element (or an ancestor/descendant of the element
        // actually hit) the topmost thing at its own center point? Without this,
        // elements hidden behind cookie banners/modals still get tagged as clickable
        // -- and across 200 different sites, some kind of overlay is close to guaranteed.
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


def safe_name(name: str) -> str:
    """Filesystem-safe version of a site name, for record/screenshot filenames."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


# --------------------------------------------------------------------------
# Site list
# --------------------------------------------------------------------------

def load_sites(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(
            f"Site list not found at {path}. Create it (see sites.json) or pass --sites <path>."
        )
    with open(path) as f:
        sites = json.load(f)

    seen = set()
    deduped = []
    for i, s in enumerate(sites):
        if "name" not in s or "url" not in s:
            log.warning(f"Entry #{i} in {path} is missing 'name' or 'url' — skipping: {s}")
            continue
        if s["name"] in seen:
            log.warning(f"Duplicate site name '{s['name']}' in config — skipping duplicate entry")
            continue
        seen.add(s["name"])
        deduped.append(s)
    return deduped


# --------------------------------------------------------------------------
# robots.txt — best-effort, fail-open (a fetch error or missing robots.txt
# means "proceed"; an explicit Disallow is respected). Cached per host so
# 200 sites on a handful of shared domains only fetch it once each.
# --------------------------------------------------------------------------

async def is_allowed(request_ctx, url: str, robots_cache: dict, robots_locks: dict) -> bool:
    host = urlparse(url).netloc
    async with robots_locks[host]:
        if host not in robots_cache:
            scheme = urlparse(url).scheme
            robots_url = f"{scheme}://{host}/robots.txt"
            parser = None
            try:
                resp = await request_ctx.get(robots_url, timeout=8000)
                if resp.ok:
                    text = await resp.text()
                    parser = RobotFileParser()
                    parser.parse(text.splitlines())
            except Exception:
                parser = None
            robots_cache[host] = parser

    parser = robots_cache[host]
    if parser is None:
        return True
    try:
        return parser.can_fetch(USER_AGENT, url)
    except Exception:
        return True


# --------------------------------------------------------------------------
# Per-site pipeline
# --------------------------------------------------------------------------

async def dismiss_consent_banners(page) -> str | None:
    """Best-effort dismissal of cookie/consent overlays so they don't occlude content."""
    for selector in CONSENT_SELECTORS:
        try:
            await page.locator(selector).first.click(timeout=1200)
            await page.wait_for_timeout(300)
            return selector
        except Exception:
            continue
    return None


async def load_and_tag(page, url: str, settle_ms: int) -> list:
    await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
    try:
        await page.wait_for_load_state("networkidle", timeout=8000)
    except PWTimeoutError:
        pass  # persistent background traffic shouldn't block the whole run
    await dismiss_consent_banners(page)
    await page.wait_for_timeout(settle_ms)
    return await page.evaluate(SOM_INJECTION_SCRIPT)


def build_records(site_name: str, screenshot_path: Path, tagged_elements: list) -> list:
    records = []
    candidates = tagged_elements.copy()
    random.shuffle(candidates)  # so the cap below is a random sample, not just the first N in DOM order

    for el in candidates[:MAX_RECORDS_PER_SITE]:
        label = (el.get("text") or "").replace("\n", " ").strip()
        if not label:
            continue

        is_input = el["tag"] in ("input", "textarea")
        if is_input:
            prompt = random.choice(PROMPT_TEMPLATES_TYPE).format(label=label)
            target = {
                "action": "type",
                "element_id": el["id"],
                "value": random.choice(TYPE_VALUES),
                "coordinate": el["center"],
            }
        else:
            prompt = random.choice(PROMPT_TEMPLATES_CLICK).format(label=label)
            target = {"action": "click", "element_id": el["id"], "coordinate": el["center"]}

        records.append({
            "site": site_name,
            "image": str(screenshot_path),
            "messages": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target)}]},
            ],
        })
    return records


async def process_site(
    browser, site: dict, sem: asyncio.Semaphore, host_sems: dict,
    robots_cache: dict, robots_locks: dict, resume: bool, ignore_robots: bool,
) -> dict:
    name, url = site["name"], site["url"]
    record_path = RECORDS_DIR / f"{safe_name(name)}.jsonl"

    if resume and record_path.exists() and record_path.stat().st_size > 0:
        count = sum(1 for _ in open(record_path))
        return {"name": name, "status": "skipped_resume", "count": count}

    host = urlparse(url).netloc
    async with sem, host_sems[host]:
        await asyncio.sleep(random.uniform(*REQUEST_JITTER_S))  # light politeness stagger

        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=USER_AGENT,
            locale="en-US",
        )
        try:
            if not ignore_robots:
                try:
                    allowed = await is_allowed(context.request, url, robots_cache, robots_locks)
                except Exception:
                    allowed = True  # never let a flaky robots.txt fetch kill the site
                if not allowed:
                    log.warning(f"[{name}] disallowed by robots.txt — skipping")
                    return {"name": name, "status": "blocked_by_robots", "count": 0}

            page = await context.new_page()
            page.set_default_timeout(NAV_TIMEOUT_MS)

            try:
                tagged = await load_and_tag(page, url, EXTRA_WAIT_MS)
            except Exception as e:
                log.warning(f"[{name}] load failed ({e}); retrying once")
                await page.wait_for_timeout(1500)
                tagged = await load_and_tag(page, url, EXTRA_WAIT_MS)

            if not tagged:
                log.info(f"[{name}] 0 elements on first pass — retrying with a longer settle time")
                tagged = await load_and_tag(page, url, RETRY_WAIT_MS)

            if not tagged:
                log.warning(f"[{name}] still 0 interactive elements — skipping")
                return {"name": name, "status": "empty", "count": 0}

            screenshot_path = IMAGES_DIR / f"{safe_name(name)}_som.png"
            await page.screenshot(path=str(screenshot_path))

            records = build_records(name, screenshot_path, tagged)
            with open(record_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            log.info(f"[{name}] tagged {len(tagged)} elements -> {len(records)} records")
            return {"name": name, "status": "success", "count": len(records)}

        except Exception as e:
            log.error(f"[{name}] failed: {e}")
            return {"name": name, "status": "failed", "count": 0, "detail": str(e)}
        finally:
            await context.close()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def merge_records() -> int:
    """Rebuild the single train_dataset.jsonl from every per-site records/*.jsonl.
    Safe to call anytime, including mid-run or after a partial crawl."""
    total = 0
    with open(FINAL_JSONL, "w") as out:
        for record_file in sorted(RECORDS_DIR.glob("*.jsonl")):
            with open(record_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")
                        total += 1
    return total


async def run(sites_path: Path, concurrency: int, force: bool, seed: int | None, ignore_robots: bool):
    if seed is not None:
        random.seed(seed)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)

    sites = load_sites(sites_path)
    log.info(f"Loaded {len(sites)} target sites from {sites_path}")

    sem = asyncio.Semaphore(concurrency)
    host_sems = defaultdict(lambda: asyncio.Semaphore(PER_HOST_CONCURRENCY))
    robots_cache: dict = {}
    robots_locks = defaultdict(asyncio.Lock)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        tasks = [
            process_site(browser, site, sem, host_sems, robots_cache, robots_locks,
                         resume=not force, ignore_robots=ignore_robots)
            for site in sites
        ]
        results = await asyncio.gather(*tasks)
        await browser.close()

    total_records = merge_records()

    by_status = defaultdict(list)
    for r in results:
        by_status[r["status"]].append(r["name"])

    log.info("=" * 60)
    log.info(f"DONE. {len(sites)} sites processed this run, {total_records} total training records -> {FINAL_JSONL}")
    for status in ("success", "empty", "blocked_by_robots", "skipped_resume", "failed"):
        names = by_status.get(status, [])
        if names:
            preview = ", ".join(names[:6]) + ("..." if len(names) > 6 else "")
            log.info(f"  {status:>16}: {len(names):>4}  [{preview}]")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Capture a Set-of-Marks UI grounding dataset across many websites")
    parser.add_argument("--sites", type=Path, default=SITES_CONFIG, help="Path to sites JSON config")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max pages in flight at once")
    parser.add_argument("--force", action="store_true", help="Ignore resume cache and reprocess every site")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible sampling")
    parser.add_argument("--ignore-robots", action="store_true", help="Skip the robots.txt check (use responsibly)")
    parser.add_argument("--merge-only", action="store_true",
                         help="Skip crawling; just rebuild train_dataset.jsonl from existing per-site records")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Where to write images/records/train_dataset.jsonl (default: dataset_som). "
                              "Use a separate dir for holdout/eval captures so they never mix with training data.")
    args = parser.parse_args()

    if args.output_dir is not None:
        global OUTPUT_DIR, IMAGES_DIR, RECORDS_DIR, FINAL_JSONL
        OUTPUT_DIR = args.output_dir
        IMAGES_DIR = OUTPUT_DIR / "images"
        RECORDS_DIR = OUTPUT_DIR / "records"
        FINAL_JSONL = OUTPUT_DIR / "train_dataset.jsonl"

    if args.merge_only:
        RECORDS_DIR.mkdir(parents=True, exist_ok=True)
        total = merge_records()
        log.info(f"Merged existing per-site records -> {total} records at {FINAL_JSONL}")
        return

    asyncio.run(run(args.sites, args.concurrency, args.force, args.seed, args.ignore_robots))


if __name__ == "__main__":
    main()

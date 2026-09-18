"""
som_common.py — single source of truth for everything that must stay
identical between capture, training, and inference: quantization config,
the SoM overlay injection script, browser/timing settings, and the
ambiguous-element dedup rule.

WHY THIS FILE EXISTS: BitsAndBytesConfig already drifted out of sync once
between train_som_qlora.py and test_agent.py/eval_offline.py (each had its
own copy, one got updated, the others didn't, and it went unnoticed until a
"fixed" eval produced suspiciously identical results). Every script that
touches the model, the browser, or the overlay marks should import from
here instead of redefining its own copy -- that turns "remember to keep N
files in sync by hand" into "there is only one copy."
"""
from __future__ import annotations

import torch
from collections import Counter
from transformers import BitsAndBytesConfig

MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
DEFAULT_ADAPTER_PATH = "./som_smolvlm_lora_adapter"

VIEWPORT = {"width": 1280, "height": 800}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 SoMDatasetBot/1.0"
)
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


def get_bnb_config() -> BitsAndBytesConfig:
    """THE quantization config, used identically for training and inference.
    Changing this here changes it everywhere -- that's the point."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["model.vision_model", "model.connector", "lm_head"],
    )


# Set-of-Marks overlay injection script. Shared by the capture pipeline AND
# the live test/eval harness so the model always sees the exact same visual
# marking style at inference that it was trained on -- if this drifted
# between the two the way bnb_config did, badge appearance itself would
# become a silent train/inference mismatch.
#
# Badge styling tuned for legibility on dense pages (protein_data_bank,
# cern_opendata were the two weakest holdout sites, both dense): larger
# font, high-contrast yellow/black, a bold border, to reduce digit
# confusion (e.g. [8] vs [3] vs [6]) in the vision encoder's downsampled
# view of small text.
#
# Anti-collision: on protein_data_bank/cern_opendata, badges were visually
# confirmed to overlap or stack directly on top of each other (dense sidebar
# nav, tightly packed link lists). Each badge now checks its position against
# every badge already placed on the page and nudges to a nearby free spot
# before being drawn -- see findFreeBadgePosition below. Falls back to the
# original default position if no free spot is found nearby, rather than
# searching indefinitely or pushing a badge off-screen.
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

    // --- Badge anti-collision additions ---
    // On dense pages (protein_data_bank, cern_opendata both confirmed visually
    // overlapping) badges were stacking directly on top of each other. Each
    // new badge now checks against every badge already placed and nudges to
    // an unoccupied spot before being drawn.
    const BADGE_PADDING_PX = 2;      // minimum gap enforced between badges
    const BADGE_HEIGHT_PX = 22;      // approx rendered badge height incl. padding/border
    const placedBadgeRects = [];     // {left, top, width, height} for every badge drawn so far

    function rectsOverlap(a, b, pad) {
        return !(
            a.left + a.width + pad <= b.left ||
            b.left + b.width + pad <= a.left ||
            a.top + a.height + pad <= b.top ||
            b.top + b.height + pad <= a.top
        );
    }

    function findFreeBadgePosition(defaultLeft, defaultTop, width, height) {
        const step = height + BADGE_PADDING_PX;
        const maxAttempts = 40; // generous headroom for long stacked columns (10+ items)

        // Cascade downward first (matches how stacked nav/sidebar items read
        // top-to-bottom), then upward, stepping one badge-height further each
        // attempt until a free slot is found -- rather than a small fixed set
        // of offsets, which runs out of options after 2-3 stacked badges and
        // silently falls back to a colliding position for everything after.
        for (const dir of [1, -1]) {
            for (let i = 0; i < maxAttempts; i++) {
                const top = defaultTop + dir * i * step;
                if (top < 0) continue;
                const rect = { left: defaultLeft, top, width, height };
                const collides = placedBadgeRects.some(p => rectsOverlap(rect, p, BADGE_PADDING_PX));
                if (!collides) return { left: defaultLeft, top };
            }
        }
        // Exhausted the vertical cascade in both directions (shouldn't happen
        // in practice) -- fall back to default rather than searching forever.
        return { left: defaultLeft, top: defaultTop };
    }

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

        const badgeText = `[${count}]`;
        const estWidth = 10 + badgeText.length * 8; // rough width estimate for collision checks
        const defaultLeft = Math.max(0, rect.left);
        const defaultTop = Math.max(0, rect.top - 20);

        const pos = findFreeBadgePosition(defaultLeft, defaultTop, estWidth, BADGE_HEIGHT_PX);
        placedBadgeRects.push({ left: pos.left, top: pos.top, width: estWidth, height: BADGE_HEIGHT_PX });

        const badge = document.createElement('div');
        badge.className = 'som-overlay-badge';
        badge.innerText = badgeText;
        badge.style.position = 'fixed';
        badge.style.left = `${pos.left}px`;
        badge.style.top = `${pos.top}px`;
        badge.style.backgroundColor = '#FF3B3B';
        badge.style.color = '#000000';
        badge.style.fontSize = '14px';
        badge.style.fontWeight = 'bold';
        badge.style.fontFamily = 'monospace';
        badge.style.padding = '2px 5px';
        badge.style.borderRadius = '3px';
        badge.style.border = '2px solid #000000';
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


def dedupe_ambiguous(elements: list) -> list:
    """Elements sharing (tag, normalized text) within one page are indistinguishable
    by any text-based instruction -- e.g. two 'Search' buttons. Training (or scoring)
    on either one encodes a contradictory mapping: same instruction, different
    "correct" element_id. Exclude every element in such a group rather than
    arbitrarily keeping one, so no ambiguous example enters the dataset at all."""
    def key(e):
        return (e["tag"], e["text"].strip().lower())
    counts = Counter(key(e) for e in elements)
    return [e for e in elements if counts[key(e)] == 1]


async def dismiss_consent_banners(page):
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
    except Exception:
        pass
    await dismiss_consent_banners(page)
    await page.wait_for_timeout(settle_ms)
    return await page.evaluate(SOM_INJECTION_SCRIPT)

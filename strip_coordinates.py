"""
strip_coordinates.py

One-time post-processing step: removes the "coordinate" field from every
assistant target in the already-scraped SoM dataset, so training targets
become {"action": ..., "element_id": ..., ["value": ...]} only.

Why: coordinate regression isn't learnable at this data scale (the eval
showed median pixel error ~410px on a 1280x800 canvas -- close to random,
with signs of memorized round numbers rather than real grounding) and it
isn't needed at inference time anyway, since the agent's own overlay-
generation code already knows every element_id's coordinate the moment it
draws the marks. Dropping it lets the model's limited capacity go entirely
toward the thing that actually matters for the demo: picking the right
element_id.

This only rewrites the per-site JSONL record files under
dataset_som/records/ -- it does NOT re-scrape or touch any screenshots.
A full backup of the untouched records is made automatically before any
in-place edit.

After running this, rebuild the merged training file with:
    python generate_som_data2.py --merge-only

Usage:
    python strip_coordinates.py                      # dataset_som/records
    python strip_coordinates.py --records-dir path/to/records
    python strip_coordinates.py --dry-run             # preview counts only, no writes
"""

import json
import shutil
import argparse
from pathlib import Path


def strip_target_coordinate(assistant_text: str) -> tuple[str, bool]:
    """Parse the assistant's JSON target string, drop 'coordinate', re-serialize.
    Returns (new_text, changed). Malformed JSON is left untouched -- fixing
    already-broken targets isn't this script's job."""
    try:
        target = json.loads(assistant_text)
    except json.JSONDecodeError:
        return assistant_text, False

    if "coordinate" not in target:
        return assistant_text, False

    del target["coordinate"]
    return json.dumps(target), True


def process_file(path: Path, dry_run: bool) -> tuple[int, int]:
    """Returns (records_seen, records_changed) for one JSONL file."""
    raw_lines = [l for l in path.read_text().splitlines() if l.strip()]
    new_lines = []
    changed_count = 0

    for line in raw_lines:
        record = json.loads(line)

        try:
            assistant_msg = next(
                m for m in record["messages"] if m["role"] == "assistant"
            )
            text_block = next(
                c for c in assistant_msg["content"] if c["type"] == "text"
            )
        except (KeyError, StopIteration):
            # Doesn't match the expected shape -- leave the record as-is.
            new_lines.append(json.dumps(record))
            continue

        new_text, changed = strip_target_coordinate(text_block["text"])
        if changed:
            text_block["text"] = new_text
            changed_count += 1

        new_lines.append(json.dumps(record))

    if not dry_run:
        path.write_text("\n".join(new_lines) + "\n")

    return len(raw_lines), changed_count


def main():
    parser = argparse.ArgumentParser(description="Strip 'coordinate' from SoM training targets")
    parser.add_argument("--records-dir", type=Path, default=Path("dataset_som/records"))
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing changes")
    args = parser.parse_args()

    if not args.records_dir.exists():
        raise FileNotFoundError(f"Records directory not found: {args.records_dir}")

    record_files = sorted(args.records_dir.glob("*.jsonl"))
    if not record_files:
        raise FileNotFoundError(f"No .jsonl files found in {args.records_dir}")

    if not args.dry_run:
        backup_dir = args.records_dir.parent / f"{args.records_dir.name}_backup_precoord"
        if backup_dir.exists():
            print(f"Backup already exists at {backup_dir} -- skipping backup "
                  f"(delete it first if you want a fresh one before rerunning).")
        else:
            shutil.copytree(args.records_dir, backup_dir)
            print(f"Backed up original records to {backup_dir}")

    total_records = 0
    total_changed = 0
    for f in record_files:
        seen, changed = process_file(f, args.dry_run)
        total_records += seen
        total_changed += changed
        if changed:
            print(f"  {f.name}: {changed}/{seen} records stripped")

    action = "Would strip" if args.dry_run else "Stripped"
    print(f"\n{action} 'coordinate' from {total_changed}/{total_records} records "
          f"across {len(record_files)} files.")

    if not args.dry_run:
        print("\nNext step, rebuild the merged training file:")
        print("    python generate_som_data2.py --merge-only")


if __name__ == "__main__":
    main()

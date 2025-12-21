import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import dataset_process as dp


def _sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return sanitized or "split"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for HH dataset processing.")
    parser.add_argument("--split", default="train[:50]", help="HF split to load.")
    parser.add_argument(
        "--output-dir",
        default="test-ouput/proccessed_data",
        help="Directory to write JSON output.",
    )
    args = parser.parse_args()

    dataset = load_dataset("Anthropic/hh-rlhf", split=args.split)
    pairs = dp._build_hh_pairs(dataset)
    if not pairs:
        raise RuntimeError("No HH pairs produced from the dataset split.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_tag = _sanitize_filename(args.split)
    output_path = output_dir / f"hh_pairs_{split_tag}.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(pairs, handle, indent=2)

    print(f"Wrote {len(pairs)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

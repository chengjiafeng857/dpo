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
    parser.add_argument("--split", default="train", help="HF split to load.")
    parser.add_argument(
        "--output-dir",
        default="test-ouput/proccessed_data",
        help="Directory to write JSON output.",
    )
    args = parser.parse_args()

    dataset = load_dataset("Anthropic/hh-rlhf", split=args.split)
    data = dp._build_hh_data(dataset, silent=False)
    if not data:
        raise RuntimeError("No HH data produced from the dataset split.")
    for prompt, info in data.items():
        responses = info.get("responses") or []
        pairs = info.get("pairs") or []
        sft_target = info.get("sft_target")
        if not responses or not pairs:
            raise RuntimeError(f"Missing responses or pairs for prompt: {prompt!r}")
        if sft_target is None:
            raise RuntimeError(f"Missing sft_target for prompt: {prompt!r}")
        if sft_target not in responses:
            raise RuntimeError(f"sft_target not in responses for prompt: {prompt!r}")
        for chosen_idx, rejected_idx in pairs:
            if chosen_idx >= len(responses) or rejected_idx >= len(responses):
                raise RuntimeError(f"Pair index out of range for prompt: {prompt!r}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_tag = _sanitize_filename(args.split)
    output_path = output_dir / f"hh_pairs_{split_tag}.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)

    print(f"Wrote {len(data)} prompts to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

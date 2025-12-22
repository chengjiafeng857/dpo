import os
import sys
from pathlib import Path

if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from sft_training import main as sft_main


def main() -> int:
    os.environ.setdefault("WANDB_MODE", "disabled")

    config_path = ROOT / "src" / "config_dpo.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["dataset"]["subset"] = "train[:5000]"
    config.setdefault("sft_training", {})["save_dir"] = "./logs/sft"

    output_dir = ROOT / "test-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_config_path = output_dir / "config_dpo.yaml"
    with test_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    original_argv = sys.argv[:]
    sys.argv = [str(SRC / "sft_training.py"), "--config", str(test_config_path)]
    try:
        sft_main()
    finally:
        sys.argv = original_argv

    print("HH SFT training test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

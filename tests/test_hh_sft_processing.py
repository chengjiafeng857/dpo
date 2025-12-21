import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from sft_training import random_controler, train_sft


def main() -> int:
    os.environ.setdefault("WANDB_MODE", "disabled")

    config_path = ROOT / "src" / "config_dpo.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["dataset"]["subset"] = "train[:500]"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_controler()

    policy_name = config["policy_name"]
    policy = AutoModelForCausalLM.from_pretrained(policy_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(policy_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy.config.pad_token_id = tokenizer.pad_token_id

    train_sft(policy, tokenizer, config, device)

    print("HH SFT training test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

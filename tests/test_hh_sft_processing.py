import os
import sys
from pathlib import Path

if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from sft_training import random_controler, train_sft
from model_utils import resolve_torch_dtype


def main() -> int:
    os.environ.setdefault("WANDB_MODE", "disabled")

    config_path = ROOT / "src" / "config_dpo.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["dataset"]["subset"] = "train[:500]"
    config.setdefault("sft_training", {})["save_dir"] = "./logs/sft"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_controler()

    policy_name = config["policy_name"]
    torch_dtype = resolve_torch_dtype(config.get("precision"))
    policy = AutoModelForCausalLM.from_pretrained(policy_name, torch_dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(policy_name)
    print(f"Loaded policy dtype: {next(policy.parameters()).dtype}")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy.config.pad_token_id = tokenizer.pad_token_id

    train_sft(policy, tokenizer, config, device)

    save_dir = ROOT / "logs" / "sft"
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("HH SFT training test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

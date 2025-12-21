import argparse
import random
import yaml
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

from dataset_process import build_sft_train_val


def load_yaml_config(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def random_controler(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def to_device_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def evaluate_sft(dataloader, policy, device, use_bf16):
    policy.eval()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = to_device_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                outputs = policy(**batch)
                loss = outputs.loss
            total_loss += loss.item()
            n += 1

    policy.train()
    return {"sft/val_loss": total_loss / max(n, 1)}


def train_sft(policy, tokenizer, config, device):
    policy.train()
    policy.requires_grad_(True)

    train_loader, val_loader = build_sft_train_val(config=config, tokenizer=tokenizer)
    optimizer = AdamW(params=policy.parameters(), lr=float(config["sft_training"]["learning_rate"]))

    use_bf16 = config["precision"] == "bf16"
    log_steps = config["sft_training"]["log_steps"]
    epochs = config["sft_training"]["epochs"]

    for epoch in range(epochs):
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"sft | epoch {epoch + 1}/{epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        running_loss = 0.0
        for step, batch in pbar:
            batch = to_device_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                outputs = policy(**batch)
                loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % log_steps == 0:
                avg_loss = running_loss / log_steps
                pbar.set_postfix(loss=f"{avg_loss:.3f}")
                if wandb.run is not None:
                    wandb.log({"sft/loss": avg_loss, "sft/epoch": epoch})
                running_loss = 0.0

        val_metrics = evaluate_sft(val_loader, policy, device, use_bf16)
        if wandb.run is not None:
            wandb.log({"sft/epoch": epoch, **val_metrics})

    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_controler()

    wandb.init(
        project=config.get("wandb_project", "handwritten-dpo"),
        name=config.get("run_name", "sft-run"),
        config=config,
    )

    policy_name = config["policy_name"]
    policy = AutoModelForCausalLM.from_pretrained(policy_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(policy_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy.config.pad_token_id = tokenizer.pad_token_id

    train_sft(policy, tokenizer, config, device)

    save_dir = config["sft_training"]["save_dir"]
    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"SFT model saved to {save_dir}")


if __name__ == "__main__":
    main()

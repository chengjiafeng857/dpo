import argparse
import random
import yaml
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

from dataset_process import build_sft_train_val
from model_utils import resolve_torch_dtype


def _torch_debug_info(device: str) -> dict:
    info = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_num_threads": torch.get_num_threads(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_device": device,
        "torch_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch_cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "torch_cudnn_deterministic": torch.backends.cudnn.deterministic,
        "torch_cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        info.update(
            {
                "cuda_device_name": props.name,
                "cuda_device_capability": f"{props.major}.{props.minor}",
                "cuda_total_memory_gb": round(props.total_memory / (1024**3), 2),
                "cuda_bf16_supported": getattr(torch.cuda, "is_bf16_supported", lambda: False)(),
            }
        )
    return info


def _log_cuda_memory(tag: str) -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        f"{tag}/cuda_mem_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 3),
        f"{tag}/cuda_mem_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 3),
        f"{tag}/cuda_max_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
    }


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
    device_type = "cuda" if device == "cuda" else "cpu"

    with torch.no_grad():
        for batch in dataloader:
            batch = to_device_batch(batch, device)
            with torch.amp.autocast(device_type=device_type, enabled=use_bf16, dtype=torch.bfloat16):
                outputs = policy(**batch)
                loss = outputs.loss
            total_loss += loss.item()
            n += 1

    policy.train()
    return {"sft/val_loss": total_loss / max(n, 1)}


def train_sft(policy, tokenizer, config, device):
    policy.train()
    policy.requires_grad_(True)
    if config.get("sft_training", {}).get("gradient_checkpointing", False):
        if hasattr(policy, "gradient_checkpointing_enable"):
            policy.gradient_checkpointing_enable()
            if getattr(policy.config, "use_cache", None) is not None:
                policy.config.use_cache = False

    train_loader, val_loader = build_sft_train_val(config=config, tokenizer=tokenizer)
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes is required for 8-bit AdamW. Install it (e.g. `uv pip install bitsandbytes`)."
        ) from exc
    optimizer = bnb.optim.AdamW8bit(
        params=policy.parameters(),
        lr=float(config["sft_training"]["learning_rate"]),
    )

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
        device_type = "cuda" if device == "cuda" else "cpu"
        running_loss = 0.0
        for step, batch in pbar:
            batch = to_device_batch(batch, device)
            with torch.amp.autocast(device_type=device_type, enabled=use_bf16, dtype=torch.bfloat16):
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
                    wandb.log(
                        {
                            "sft/loss": avg_loss,
                            "sft/epoch": epoch,
                            **_log_cuda_memory("sft/train"),
                        }
                    )
                running_loss = 0.0

        val_metrics = evaluate_sft(val_loader, policy, device, use_bf16)
        if wandb.run is not None:
            wandb.log({"sft/epoch": epoch, **val_metrics, **_log_cuda_memory("sft/val")})
        else:
            print(
                f"sft | epoch {epoch + 1}/{epochs} val_loss={val_metrics['sft/val_loss']:.4f}"
            )

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
    torch_dtype = resolve_torch_dtype(config.get("precision"))
    policy = AutoModelForCausalLM.from_pretrained(policy_name, torch_dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(policy_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy.config.pad_token_id = tokenizer.pad_token_id

    if wandb.run is not None:
        debug_info = _torch_debug_info(device)
        wandb.config.update(debug_info, allow_val_change=True)
        watch_log = config.get("sft_training", {}).get("watch_log", "gradients")
        watch_freq = config.get("sft_training", {}).get("watch_log_freq", 100)
        wandb.watch(policy, log=watch_log, log_freq=watch_freq)

    train_sft(policy, tokenizer, config, device)

    save_dir = config["sft_training"]["save_dir"]
    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"SFT model saved to {save_dir}")


if __name__ == "__main__":
    main()

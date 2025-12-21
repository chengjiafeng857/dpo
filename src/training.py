import torch
from torch.optim import AdamW
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from dataset_process import build_train_val
from sft_training import train_sft
from dpo_loss import dpo_loss
from batch_log_prob import compute_batch_log_prob
import wandb

# load yaml config
def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
# transform the batch to the device
def to_device_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

def evaluate(dataloader, policy, ref_model, device, config, use_bf16):
    policy.eval(); ref_model.eval()

    total_loss = 0.0
    total_acc  = 0.0
    total_margin = 0.0
    n = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = to_device_batch(batch, device)
            
            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                (p_ch, p_rj, r_ch, r_rj) = compute_batch_log_prob(batch, policy, ref_model)
                
                loss, ch_lp, rj_lp, acc, margin = dpo_loss(
                    policy_chosen_log_prob=p_ch,
                    policy_rejected_log_prob=p_rj,
                    ref_chosen_log_prob=r_ch,
                    ref_rejected_log_prob=r_rj,
                    beta=config['dpo_training']['beta']
                    )
                
            total_loss   += loss.item()
            total_acc    += acc.item()
            total_margin += margin.item()
            n += 1

    policy.train()
    return {
        "val/loss":   total_loss / max(n, 1),
        "val/acc":    total_acc / max(n, 1),
        "val/margin": total_margin / max(n, 1),
    }
    
# control randomness
def random_controler(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args=parser.parse_args()

    config = load_yaml_config(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    random_controler()

    # initial wandb
    wandb.init(project=config.get('wandb_project','handwritten-dpo'),
               name=config.get('run_name','run'),
               config=config)

    # load model and tokenizer
    policy_name = config["policy_name"]
    policy = AutoModelForCausalLM.from_pretrained(policy_name).to(device)
    tok = AutoTokenizer.from_pretrained(policy_name)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    policy.config.pad_token_id = tok.pad_token_id

    sft_config = config.get("sft_training")
    run_sft = bool(sft_config and sft_config.get("enabled", True))
    if run_sft:
        print("Running SFT before DPO training...")
        train_sft(policy, tok, config, device)
        sft_save_dir = sft_config.get("save_dir", "sft_model")
        policy.save_pretrained(sft_save_dir)
        tok.save_pretrained(sft_save_dir)
        ref_name = sft_save_dir
    else:
        ref_name = config["ref_name"]

    ref_model = AutoModelForCausalLM.from_pretrained(ref_name).to(device)
    ref_model.config.pad_token_id = tok.pad_token_id
    ref_model.requires_grad_(False)
    
    # load dataset
    train_loader, val_loader = build_train_val(config=config, tokenizer=tok)

    policy.train()
    ref_model.eval()

    # define optimizer
    optimizer = AdamW(params=policy.parameters(), lr=float(config['dpo_training']['learning_rate']))

    # using bf 16
    use_bf16 = config['precision'] == 'bf16'

    log_steps = config['dpo_training']['log_steps']
    ref_model.to(dtype=torch.bfloat16)
    
    # training loop
    epochs = config['dpo_training']['epochs']
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader),
                total=len(train_loader),
                desc=f"train | epoch {epoch+1}/{epochs}",
                dynamic_ncols=True, leave=False)
        
        running_loss = 0.0
        for step, batch in pbar:
            batch = to_device_batch(batch, device)

            # generate logits for each part
            # using bf16
            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16): 
                # compute log_prob
                policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob = compute_batch_log_prob(batch, policy=policy, ref_model=ref_model)

                # compute loss
                loss, chosen_log_prob, rejected_log_prob, reward_accuracy, reward_margins = dpo_loss(
                     policy_chosen_log_prob=policy_chosen_log_prob,
                     policy_rejected_log_prob=policy_rejected_log_prob,
                     ref_chosen_log_prob=ref_chosen_log_prob,
                     ref_rejected_log_prob=ref_rejected_log_prob,
                     beta=config['dpo_training']['beta']
                     )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # log the training info
            if (step + 1) % log_steps == 0:
                avg_loss = running_loss / log_steps
                pbar.set_postfix(loss=f"{avg_loss:.3f}", reward_accuracy=f"{reward_accuracy.item():.2f}", reward_margins=f"{reward_margins.item():.3f}")
                wandb.log({
                    'loss': avg_loss,
                    'chosen_log_prob': chosen_log_prob.item(),
                    'rejected_log_prob': rejected_log_prob.item(),
                    'reward_accuracy': reward_accuracy.item(),
                    'reward_margins': reward_margins.item()
                })
                running_loss = 0.0

        val_metrics = evaluate(val_loader, policy, ref_model, device, config, use_bf16)
        wandb.log({"epoch": epoch, **val_metrics})

    # save model
    policy.save_pretrained("dpo_model")

if __name__ == "__main__":
    train()

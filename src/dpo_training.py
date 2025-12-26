import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
import copy
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
# Import our new unified dataloader
from dataloader import UnifiedHHRLHFDataset, unified_collate_fn

import argparse
import os

# ==========================================
# Configuration (Defaults)
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="DPO Training Script")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model (or path to SFT model)")
    parser.add_argument("--offset", type=int, default=1000, help="Number of samples")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO Beta")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="dpo-pytorch-demo", help="WandB project name")
    parser.add_argument("--output_dir", type=str, default="dpo_model_final", help="Directory to save the model")
    return parser.parse_args()

# Set device (Mac usage prioritized for this user)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Dummy Dataset & Processing
# ==========================================
# ==========================================
# 1. Dataset Loading (Anthropic/hh-rlhf)
# ==========================================
def load_hh_dataset(split="train", num_samples=1000):
    """
    Loads the Anthropic/hh-rlhf dataset and converts it to a list of dicts.
    """
    print(f"Loading Anthropic/hh-rlhf dataset ({split})...")
    # Using a small subset or streaming could be faster for debugging, 
    # but we'll load the split as requested.
    ds = load_dataset("Anthropic/hh-rlhf", split=split)
    
    # Convert to list of dicts for our custom Dataset class
    # The dataset has 'chosen' and 'rejected' columns already.
    data_list = []
    # Limit for demonstration purposes if needed, effectively "streaming" a small buffer
    # But for real training we'd use the whole thing. 
    # Let's take 1000 samples for this demo script to run quickly.
    # User can remove the slice to use full dataset.
    print(f"Selecting first {num_samples} samples for demo purposes...")
    for item in ds.select(range(num_samples)):
        data_list.append({
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        })
    return data_list

# ==========================================
# 2. Helper Functions (Math & Logic)
# ==========================================
def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """
    Computes the log probabilities of the given labels under the logits.
    
    Args:
        logits: Shape (batch_size, sequence_len, vocab_size)
        labels: Shape (batch_size, sequence_len) - contains -100 for ignored tokens (prompts)
    
    Returns:
        log_prob: Shape (batch_size,) - Sum of log probabilities for the completion tokens
    """
    # 1. Check shapes to ensure correctness
    assert logits.shape[:-1] == labels.shape

    # 2. Shift labels and logits so that the token at index `t` predicts the token at index `t+1`.
    #    Discard the last logit (prediction for t+1 which has no target)
    #    Discard the first label (the token at t=0, which has no history to predict it from)
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    # 3. Create a mask to filter out padding/-100 labels
    loss_mask = (labels != -100)

    # 4. Dummy labels for gather (replace -100 with 0 so gather doesn't crash)
    #    The mask will ensure we zero out their contribution later.
    labels[labels == -100] = 0

    # 5. Compute log_softmax across vocabulary
    #    per_token_logps: (batch, seq_len)
    dist = F.log_softmax(logits, dim=-1)

    # 6. Gather the log probability of the actual target token for each position
    #    gather dim=2 means we select the log_prob of the specific index in `labels`
    per_token_logps = torch.gather(dist, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    # 7. Zero out probabilities for masked tokens (prompts and padding)
    per_token_logps = per_token_logps * loss_mask

    # 8. Sum across the sequence to get total log probability of the completion
    return per_token_logps.sum(-1)

def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    """
    The Core DPO Loss Calculation.

    Formula:
    L_DPO = -E [ log sigma ( beta * (log(pi(yw|x)/pi_ref(yw|x)) - log(pi(yl|x)/pi_ref(yl|x))) ) ]

    Explanation:
    1. We calculate the log-ratio of the policy vs reference for the chosen response (yw):
       log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
    
    2. We calculate the log-ratio of the policy vs reference for the rejected response (yl):
       log_ratio_rejected = policy_rejected_logps - ref_rejected_logps
       
    3. The difference between these ratios represents how much more likelihood mass 
       the policy assigns to the chosen vs rejected, relative to the reference model.
       logits = log_ratio_chosen - log_ratio_rejected
       
    4. We scale this by 'beta'. A higher beta means the reference model acts as a stronger anchor,
       and the policy is penalized more for deviating unless it's very confident.
       scaled_logits = beta * logits
       
    5. We minimize the negative log likelihood of the sigmoid of this value.
       This is equivalent to maximizing the probability that the chosen score > rejected score.
    """
    
    # pi_logratios = policy_logps - ref_logps
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    # The 'logits' input to the sigmoid
    # We want to maximize: sigmoid(beta * (pi_diff - ref_diff))
    logits = pi_logratios - ref_logratios
    
    losses = -F.logsigmoid(beta * logits)
    
    # Also calculate standard 'chosen' and 'rejected' rewards for logging
    # The implicit reward in DPO is: beta * (log pi(y|x) - log ref(y|x))
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

# ==========================================
# 3. Main Training Routine
# ==========================================
def train():
    args = parse_args()
    wandb.init(project=args.wandb_project, config=args)
    torch.manual_seed(args.seed)

    print("Loading models...")
    # Load Pretrained Model (Policy)
    print(f"Loading model from: {args.model_name}")
    policy_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # GPT2 has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create Reference Model (Frozen copy of Policy)
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    ref_model.requires_grad_(False) # Freeze reference model completely

    # Optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr)

    # DataLoader
    print("Preparing Dataset...")
    train_data_list = load_hh_dataset("train", num_samples=args.offset)
    
    dataset = UnifiedHHRLHFDataset(
        data=train_data_list, 
        tokenizer=tokenizer, 
        mode='dpo'
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: unified_collate_fn(b, tokenizer)
    )

    print("Starting Training Loop...")
    
    # List to store logits from every step
    all_step_logits = []
    
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            policy_model.train()
            
            # Move batch to device
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            # ----------------------------------------------------------------
            # FORWARD PASS - Policy Model
            # ----------------------------------------------------------------
            # Get logits for chosen and rejected
            chosen_logits = policy_model(chosen_ids, attention_mask=chosen_mask).logits
            rejected_logits = policy_model(rejected_ids, attention_mask=rejected_mask).logits
            
            # Calculate log probabilities sum for the completion
            policy_chosen_logps = get_batch_logps(chosen_logits, chosen_labels)
            policy_rejected_logps = get_batch_logps(rejected_logits, rejected_labels)

            # Store the logits for analysis
            # WARNING: Collecting all logits can consume large amounts of memory for long training runs
            all_step_logits.append({
                "epoch": epoch,
                "step": step,
                "chosen_logits": chosen_logits.detach().cpu(),
                "rejected_logits": rejected_logits.detach().cpu()
            })

            # ----------------------------------------------------------------
            # FORWARD PASS - Reference Model (No Grad)
            # ----------------------------------------------------------------
            with torch.no_grad():
                ref_chosen_logits = ref_model(chosen_ids, attention_mask=chosen_mask).logits
                ref_rejected_logits = ref_model(rejected_ids, attention_mask=rejected_mask).logits
                
                ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_labels)
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_labels)

            # ----------------------------------------------------------------
            # LOSS CALCULATION
            # ----------------------------------------------------------------
            loss, reward_chosen, reward_rejected = dpo_loss(
                policy_chosen_logps, 
                policy_rejected_logps, 
                ref_chosen_logps, 
                ref_rejected_logps, 
                beta=args.beta
            )

            # ----------------------------------------------------------------
            # BACKWARD PASS
            # ----------------------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ----------------------------------------------------------------
            # LOGGING
            # ----------------------------------------------------------------
            # Helper metric: "Reward Margin" = chosen_reward - rejected_reward
            reward_margin = reward_chosen - reward_rejected

            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | Margin: {reward_margin.item():.4f}")
            
            wandb.log({
                "loss": loss.item(),
                "reward_chosen": reward_chosen.item(),
                "reward_rejected": reward_rejected.item(),
                "reward_margin": reward_margin.item(),
                "policy_chosen_logps": policy_chosen_logps.mean().item(),
                "policy_rejected_logps": policy_rejected_logps.mean().item()
            })

    print("Training Complete.")
    
    # Save the collected logits to disk
    print(f"Saving collected logits to dpo_logits.pt...")
    torch.save(all_step_logits, "dpo_logits.pt")

    # Save final model
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    policy_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    print("Done.")

    wandb.finish()

if __name__ == "__main__":
    train()

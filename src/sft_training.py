import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import wandb
import os

# Import our unified dataloader
from dataloader import UnifiedHHRLHFDataset, unified_collate_fn

# ==========================================
# Configuration
# ==========================================
import argparse

# ==========================================
# Configuration (Defaults)
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training Script")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model to fine-tune")
    parser.add_argument("--offset", type=int, default=1000, help="Number of samples to use")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="sft-pytorch-demo", help="WandB project name")
    parser.add_argument("--output_dir", type=str, default="sft_model_final", help="Directory to save the model")
    return parser.parse_args()

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_hh_dataset_sft(split="train", num_samples=1000):
    """
    Loads the Anthropic/hh-rlhf dataset and converts it to list of dicts.
    """
    print(f"Loading Anthropic/hh-rlhf dataset ({split})...")
    ds = load_dataset("Anthropic/hh-rlhf", split=split)
    
    data_list = []
    # Limit for demo purposes
    print(f"Selecting first {num_samples} samples for SFT demo...")
    for item in ds.select(range(num_samples)):
        data_list.append({
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        })
    return data_list

def train():
    args = parse_args()
    wandb.init(project=args.wandb_project, config=args)
    torch.manual_seed(args.seed)

    print(f"Loading model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)

    print("Preparing Dataset...")
    train_data = load_hh_dataset_sft("train", num_samples=args.offset)
    
    # Instantiate Dataset in SFT mode
    dataset = UnifiedHHRLHFDataset(
        data=train_data, 
        tokenizer=tokenizer, 
        mode='sft' 
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: unified_collate_fn(b, tokenizer)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    print("Starting SFT Training...")
    model.train()
    
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                wandb.log({"loss": loss.item()})
                
    print("Training Complete. Saving model...")
    # Save the final model
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    
    wandb.finish()

if __name__ == "__main__":
    train()

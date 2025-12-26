import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Any

# ==========================================
# 1. Parsing Logic
# ==========================================
def extract_prompt_and_response(text: str) -> Tuple[str, str]:
    """
    Parses the Anthropic HH-RLHF format.
    Format is typically: "\n\nHuman: <prompt>\n\nAssistant: <response>"
    
    We split on the LAST occurrence of "\n\nAssistant:" to handle multi-turn dialogues correctly.
    Everything before the last "\n\nAssistant:" (including the tag itself) is the prompt.
    Everything after is the response.
    """
    search_term = "\n\nAssistant:"
    split_idx = text.rfind(search_term)
    
    if split_idx == -1:
        # Fallback if format is unexpected (should warn in real scenario)
        return text, ""
    
    # Include the "Assistant:" tag in the prompt so the model prompts completion
    prompt = text[:split_idx + len(search_term)]
    response = text[split_idx + len(search_term):]
    
    return prompt, response


# ==========================================
# 2. Unified Dataset Class
# ==========================================
class UnifiedHHRLHFDataset(Dataset):
    """
    A unified Dataset class for Anthropic/hh-rlhf style data.
    Supports two modes:
    1. mode='sft': Returns tokenized inputs + labels for Supervised Fine-Tuning.
                   Labels for the prompt part are masked (-100).
    2. mode='dpo': Returns tokenized 'chosen' and 'rejected' responses + prompts.
                   Used for Direct Preference Optimization.
    """
    def __init__(
        self, 
        data: List[Dict[str, str]], 
        tokenizer: PreTrainedTokenizerBase, 
        mode: str = 'sft',
        max_length: int = 1024
    ):
        """
        Args:
            data: List of dicts, each containing 'chosen' and 'rejected' keys (HF dataset format).
            tokenizer: HuggingFace tokenizer.
            mode: 'sft' or 'dpo'.
            max_length: Maximum sequence length.
        """
        assert mode in ['sft', 'dpo'], "mode must be either 'sft' or 'dpo'"
        self.data = data
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        
        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Add unique pad token if absolutely necessary, but usually EOS works for GPT-2
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # We always extract from the 'chosen' field for SFT.
        # For DPO, we need both.
        # HH-RLHF 'chosen'/'rejected' fields contain the FULL dialogue (Prompt + Response).
        
        # 1. Parse texts
        # Note: HH dataset format has 'chosen' = Prompt + Response
        # We assume 'chosen' and 'rejected' share the same Prompt.
        prompt_text, chosen_resp_text = extract_prompt_and_response(item['chosen'])
        _, rejected_resp_text = extract_prompt_and_response(item['rejected'])
        
        if self.mode == 'sft':
            return self._prepare_sft_sample(prompt_text, chosen_resp_text)
        else:
            return self._prepare_dpo_sample(prompt_text, chosen_resp_text, rejected_resp_text)

    def _prepare_sft_sample(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        """
        Prepares a single sample for SFT.
        Returns:
            input_ids: [Prompt Tokens + Response Tokens]
            labels:    [-100... (Prompt) ... -100 + Response Tokens]
            attention_mask: 1s for all
        """
        # Tokenize parts separately to know lengths
        # add_special_tokens=False to control special tokens manually if needed
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        
        # Add EOS to response if not present (good practice for generation)
        if self.tokenizer.eos_token_id is not None:
            response_ids.append(self.tokenizer.eos_token_id)

        # Concatenate
        input_ids = prompt_ids + response_ids
        
        # Create labels: mask prompt with -100
        labels = [-100] * len(prompt_ids) + response_ids
        
        # Truncate if necessary
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            # Attention mask is handled in collate based on padding
            # But we return the length here conceptually
            "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long) 
        }

    def _prepare_dpo_sample(self, prompt: str, chosen: str, rejected: str) -> Dict[str, Any]:
        """
        Prepares a single sample for DPO.
        Retuns raw lists/tensors that can be padded in collate.
        We return (prompt + chosen) and (prompt + rejected).
        """
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        chosen_ids = self.tokenizer.encode(chosen, add_special_tokens=False)
        rejected_ids = self.tokenizer.encode(rejected, add_special_tokens=False)
        
        if self.tokenizer.eos_token_id is not None:
            chosen_ids.append(self.tokenizer.eos_token_id)
            rejected_ids.append(self.tokenizer.eos_token_id)

        # Build full sequences
        chosen_input_ids = prompt_ids + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        
        # Create labels for DPO (mask prompt)
        # We need these to calculate logprobs of ONLY the completion part
        chosen_labels = [-100] * len(prompt_ids) + chosen_ids
        rejected_labels = [-100] * len(prompt_ids) + rejected_ids

        # Truncate
        def truncate(ids, lbs):
            if len(ids) > self.max_length:
                return ids[:self.max_length], lbs[:self.max_length]
            return ids, lbs
            
        chosen_input_ids, chosen_labels = truncate(chosen_input_ids, chosen_labels)
        rejected_input_ids, rejected_labels = truncate(rejected_input_ids, rejected_labels)

        return {
            # We explicitly separate chosen and rejected
            "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
            "prompt_text": prompt  # Useful for debugging
        }


# ==========================================
# 3. Custom Collate Function
# ==========================================
def unified_collate_fn(batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase):
    """
    Handles dynamic padding for the batch.
    Detects mode based on keys present in the first item.
    """
    if len(batch) == 0:
        return {}
    
    # Check mode
    is_dpo = "chosen_input_ids" in batch[0]
    pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    if not is_dpo:
        # --- SFT COLLATING ---
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Dynamic padding (batch_first=True)
        # Input IDs pad with pad_token_id
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_val)
        
        # Labels pad with -100 (ignore index)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        # Create attention mask (1 where not pad, 0 where pad)
        # Note: We reconstruct mask from padded input_ids because we want 0 on the pads
        attention_mask = (input_ids_padded != pad_val).long()
        
        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask
        }
        
    else:
        # --- DPO COLLATING ---
        chosen_ids = [item['chosen_input_ids'] for item in batch]
        chosen_labels = [item['chosen_labels'] for item in batch]
        rejected_ids = [item['rejected_input_ids'] for item in batch]
        rejected_labels = [item['rejected_labels'] for item in batch]
        
        # Pad everything
        c_ids_padded = torch.nn.utils.rnn.pad_sequence(chosen_ids, batch_first=True, padding_value=pad_val)
        c_labels_padded = torch.nn.utils.rnn.pad_sequence(chosen_labels, batch_first=True, padding_value=-100)
        c_mask = (c_ids_padded != pad_val).long()
        
        r_ids_padded = torch.nn.utils.rnn.pad_sequence(rejected_ids, batch_first=True, padding_value=pad_val)
        r_labels_padded = torch.nn.utils.rnn.pad_sequence(rejected_labels, batch_first=True, padding_value=-100)
        r_mask = (r_ids_padded != pad_val).long()

        return {
            "chosen_input_ids": c_ids_padded,
            "chosen_labels": c_labels_padded,
            "chosen_attention_mask": c_mask,
            "rejected_input_ids": r_ids_padded,
            "rejected_labels": r_labels_padded,
            "rejected_attention_mask": r_mask
        }

# ==========================================
# 4. Verification / Example Usage
# ==========================================
if __name__ == "__main__":
    print("Running Dataloader Verification...")
    
    # 1. Setup Dummy Tokenizer
    # We use gpt2 as it's small and standard.
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token # GPT2 fix
    except:
        print("Could not load gpt2 tokenizer. Ensure transformers is installed.")
        exit(1)

    # 2. Dummy Data (HH-RLHF format)
    dummy_data = [
        {
            "chosen": "\n\nHuman: What is AI?\n\nAssistant: AI stands for Artificial Intelligence.",
            "rejected": "\n\nHuman: What is AI?\n\nAssistant: It is a type of food."
        },
        {
            "chosen": "\n\nHuman: Help me code.\n\nAssistant: Sure, here is python code.",
            "rejected": "\n\nHuman: Help me code.\n\nAssistant: I cannot help with that."
        }
    ]
    
    # 3. Verify SFT Mode
    print("\n--- Testing SFT Mode ---")
    sft_ds = UnifiedHHRLHFDataset(dummy_data, tokenizer, mode='sft')
    sft_loader = DataLoader(sft_ds, batch_size=2, collate_fn=lambda x: unified_collate_fn(x, tokenizer))
    
    for batch in sft_loader:
        print("Batch Keys:", batch.keys())
        print("Input IDs Shape:", batch['input_ids'].shape)
        print("Labels Shape:", batch['labels'].shape)
        print("Attention Mask Shape:", batch['attention_mask'].shape)
        
        # Check masking: The first few tokens (Prompt) should be -100 in labels
        # Let's inspect the first item
        prompt_len_approx = len(tokenizer.encode("\n\nHuman: What is AI?\n\nAssistant:"))
        print(f"Sample Label Head (should be -100): {batch['labels'][0][:5]}")
        break  # Only need one batch
        
    # 4. Verify DPO Mode
    print("\n--- Testing DPO Mode ---")
    dpo_ds = UnifiedHHRLHFDataset(dummy_data, tokenizer, mode='dpo')
    dpo_loader = DataLoader(dpo_ds, batch_size=2, collate_fn=lambda x: unified_collate_fn(x, tokenizer))
    
    for batch in dpo_loader:
        print("Batch Keys:", batch.keys())
        print("Chosen IDs Shape:", batch['chosen_input_ids'].shape)
        print("Rejected IDs Shape:", batch['rejected_input_ids'].shape)
        
        # Check logic: chosen and rejected should have same prompt prefix
        # We can check specific tokens if we really want, but shape check is good first step
        break
        
    print("\nVerification Complete. Script is ready.")

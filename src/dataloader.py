import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Any

# ==========================================
# 1. Parsing Logic (Source Data Specific)
# ==========================================
def extract_prompt_and_response(text: str) -> Tuple[str, str]:
    """
    Parses the Anthropic HH-RLHF format.
    Format is typically: "\n\nHuman: <prompt>\n\nAssistant: <response>"
    
    We split on the LAST occurrence of "\n\nAssistant:" to handle multi-turn dialogues.
    We clean the tags to extract the raw content for chat templates.
    """
    search_term = "\n\nAssistant:"
    split_idx = text.rfind(search_term)
    
    if split_idx == -1:
        return text, ""
    
    # Extract raw content
    # Prompt: Everything up to the last Assistant tag
    # We also need to strip the leading "\n\nHuman:" if possible to get clean content
    prompt_raw = text[:split_idx]
    response_raw = text[split_idx + len(search_term):]
    
    # Cleanup Human tag
    human_tag = "\n\nHuman:"
    if prompt_raw.startswith(human_tag):
        prompt_raw = prompt_raw[len(human_tag):]
    elif prompt_raw.startswith("Human:"): # Sometimes start of file
        prompt_raw = prompt_raw[len("Human:"):]
        
    return prompt_raw.strip(), response_raw.strip()


# ==========================================
# 2. Unified Dataset Class
# ==========================================
class UnifiedHHRLHFDataset(Dataset):
    """
    A unified Dataset class for Anthropic/hh-rlhf style data.
    Uses generic Chat Templates for tokenization.
    """
    def __init__(
        self, 
        data: List[Dict[str, str]], 
        tokenizer: PreTrainedTokenizerBase, 
        mode: str = 'sft',
        max_length: int = 1024
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Extract raw content (clean text)
        prompt_text, chosen_resp_text = extract_prompt_and_response(item['chosen'])
        _, rejected_resp_text = extract_prompt_and_response(item['rejected'])
        
        if self.mode == 'sft':
            return self._prepare_sft_sample(prompt_text, chosen_resp_text)
        else:
            return self._prepare_dpo_sample(prompt_text, chosen_resp_text, rejected_resp_text)

    def _prepare_sft_sample(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        # Construct Chat Messages
        # Note: We treat the entire history as "user" here for simplicity if it was multi-turn,
        # or we could try to parse turns. For HH-RLHF, typically prompt is the context.
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        
        # Tokenize entire dialogue
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False, # We want the full conversation including assistant response
            truncation=True, 
            max_length=self.max_length
        )
        
        # Determine where to mask. Tokenize just the prompt (with generation prompt)
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=True, 
            add_generation_prompt=True, # Include assistant header
        )
        
        # Create labels
        # Mask prompt tokens. The boundary is len(prompt_ids).
        # Note: If input_ids was truncated differently than prompt_ids, this could be risky,
        # but usually prompt is shorter than max_len.
        
        labels = [-100] * len(input_ids)
        
        # Only assign labels to the response part
        if len(prompt_ids) < len(input_ids):
            for i in range(len(prompt_ids), len(input_ids)):
                labels[i] = input_ids[i]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long) 
        }

    def _prepare_dpo_sample(self, prompt: str, chosen: str, rejected: str) -> Dict[str, Any]:
        # 1. Prompt Only (for finding length)
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=True, 
            add_generation_prompt=True
        )
        
        # 2. Chosen
        chosen_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen}
        ]
        chosen_input_ids = self.tokenizer.apply_chat_template(
            chosen_messages, tokenize=True, truncation=True, max_length=self.max_length
        )
        
        # 3. Rejected
        rejected_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected}
        ]
        rejected_input_ids = self.tokenizer.apply_chat_template(
            rejected_messages, tokenize=True, truncation=True, max_length=self.max_length
        )
        
        # Create labels (mask prompt)
        chosen_labels = [-100] * len(chosen_input_ids)
        if len(prompt_ids) < len(chosen_input_ids):
            for i in range(len(prompt_ids), len(chosen_input_ids)):
                chosen_labels[i] = chosen_input_ids[i]
                
        rejected_labels = [-100] * len(rejected_input_ids)
        if len(prompt_ids) < len(rejected_input_ids):
            for i in range(len(prompt_ids), len(rejected_input_ids)):
                rejected_labels[i] = rejected_input_ids[i]

        return {
            "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
            "prompt_text": prompt
        }


# ==========================================
# 3. Custom Collate Function
# ==========================================
def unified_collate_fn(batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase):
    if len(batch) == 0:
        return {}
    
    is_dpo = "chosen_input_ids" in batch[0]
    pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    if not is_dpo:
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_val)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = (input_ids_padded != pad_val).long()
        
        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask
        }
    else:
        chosen_ids = [item['chosen_input_ids'] for item in batch]
        chosen_labels = [item['chosen_labels'] for item in batch]
        rejected_ids = [item['rejected_input_ids'] for item in batch]
        rejected_labels = [item['rejected_labels'] for item in batch]
        
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
# 4. Verification Check
# ==========================================
if __name__ == "__main__":
    print("Running Chat-Template Dataloader Verification...")
    try:
        # We try to load Qwen tokenizer to test verify it works, or fallback
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
        print("Loaded Qwen tokenizer.")
    except:
        print("Qwen tokenizer not found or failed. Falling back to GPT2 for logic check.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
    dummy_data = [
        {
            "chosen": "\n\nHuman: What is AI?\n\nAssistant: AI is artificial intelligence.",
            "rejected": "\n\nHuman: What is AI?\n\nAssistant: Food."
        }
    ]
    
    ds = UnifiedHHRLHFDataset(dummy_data, tokenizer, mode='sft')
    item = ds[0]
    print("SFT Item Keys:", item.keys())
    print("SFT Input IDs:", item['input_ids'])
    print("SFT Labels:", item['labels'])
    
    dpo_ds = UnifiedHHRLHFDataset(dummy_data, tokenizer, mode='dpo')
    dpo_item = dpo_ds[0]
    print("DPO Item Keys:", dpo_item.keys())
    print("Verified.")

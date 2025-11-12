import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from functools import partial

# process prompt-choosen-rejected pairs
def process_ds(batch, tokenizer, max_len):
    chosen, rejected, prompt_length = [], [], []
    for item in batch:
        # extract prompt and response
        prompt_txt   = "Prompt: "   + item["question"].strip()   + "\n"
        chosen_txt   = "Response: " + item["chosen"].strip()   + "\n"
        rejected_txt = "Response: " + item["rejected"].strip() + "\n"

        # claculate the original prompt length
        # tokenizer will return a dit, we only need input_ids tensor
        prompt_ids = tokenizer(
            prompt_txt, 
            add_special_tokens=False, 
            return_tensors=None
        )["input_ids"]

        prompt_length.append(len(prompt_ids))

        # combine prompt+response
        chosen.append(prompt_txt + chosen_txt)
        rejected.append(prompt_txt + rejected_txt)

    # tokenizer    
    enc_chosen = tokenizer(
        chosen,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=True,  
    )
    enc_rejected = tokenizer(
        rejected,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=True,
    )

    # process bos token for computing prompt length
    has_bos = (
        tokenizer.bos_token_id is not None
        and enc_chosen.input_ids.shape[1] > 0
        and enc_chosen.input_ids[0, 0].item() == tokenizer.bos_token_id
    )
    bos_shift = 1 if has_bos else 0

    # prompt length
    prompt_length = torch.tensor(
        [pl + bos_shift for pl in prompt_length], dtype=torch.long
    )

    out_batch = {
        "prompt_chosen_ids":  enc_chosen.input_ids,          
        "prompt_chosen_mask": enc_chosen.attention_mask,     
        "prompt_rejected_ids":  enc_rejected.input_ids,      
        "prompt_rejected_mask": enc_rejected.attention_mask, 
        "prompt_length": prompt_length,                     
    }

    return out_batch

def build_train_val(config, tokenizer):
    dataset_name = config['dataset']['dataset_name']
    ds_raw = load_dataset(dataset_name, split=config['dataset']['subset'])
    if "validation" in ds_raw:
        train_ds_raw = ds_raw["train"]
        val_ds_raw = ds_raw["validation"]
    else:
        split = ds_raw.train_test_split(test_size=config['dataset']['val_ratio'], seed=config['dataset']['seed'])
        train_ds_raw, val_ds_raw = split["train"], split["test"]

    ds_collate = partial(process_ds, tokenizer=tokenizer, max_len=config["dataset"]["max_len"])
    batch_size = config["dpo_training"]["batch_size"]

    train_loader = DataLoader(
        train_ds_raw,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ds_collate,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds_raw,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ds_collate,
        pin_memory=True,
    )

    return train_loader, val_loader



    
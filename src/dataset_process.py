import torch
from collections import defaultdict
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from functools import partial
from typing import Dict, Iterable, List, Tuple


def extract_anthropic_prompt(prompt_and_response: str) -> str:
    """Extract the Anthropic prompt from a prompt + response string."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    if search_term_idx == -1:
        raise ValueError(f"Prompt does not contain '{search_term}'")
    return prompt_and_response[:search_term_idx + len(search_term)]


def _is_hh_dataset(name: str) -> bool:
    name = name.lower()
    return name in {"hh", "hh-rlhf", "anthropic/hh-rlhf"}


def _is_shp_dataset(name: str) -> bool:
    name = name.lower()
    return name in {"shp", "stanfordnlp/shp"}


def _flatten_prompt_pairs(data: Dict[str, Dict[str, List]]) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    for prompt, info in data.items():
        responses = info["responses"]
        for chosen_idx, rejected_idx in info["pairs"]:
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": responses[chosen_idx],
                    "rejected": responses[rejected_idx],
                }
            )
    return pairs


def _build_hh_pairs(dataset: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    data: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for row in dataset:
        prompt = extract_anthropic_prompt(row["chosen"])
        chosen_response = row["chosen"][len(prompt):]
        rejected_response = row["rejected"][len(prompt):]

        n_responses = len(data[prompt]["responses"])
        data[prompt]["pairs"].append((n_responses, n_responses + 1))
        data[prompt]["responses"].extend([chosen_response, rejected_response])

    return _flatten_prompt_pairs(data)


def _build_shp_pairs(dataset: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    data: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for row in dataset:
        prompt = "\n\nHuman: " + row["history"] + "\n\nAssistant:"
        responses = [" " + row["human_ref_A"], " " + row["human_ref_B"]]
        scores = [row["score_A"], row["score_B"]]

        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        n_responses = len(data[prompt]["responses"])
        chosen_first = row["labels"] == 1
        data[prompt]["pairs"].append((n_responses, n_responses + 1) if chosen_first else (n_responses + 1, n_responses))
        data[prompt]["responses"].extend(responses)

    return _flatten_prompt_pairs(data)


def _extract_texts(item: Dict[str, str]) -> Tuple[str, str, str]:
    if "prompt" in item:
        return item["prompt"], item["chosen"], item["rejected"]

    prompt_txt = "Prompt: " + item["question"].strip() + "\n"
    chosen_txt = "Response: " + item["chosen"].strip() + "\n"
    rejected_txt = "Response: " + item["rejected"].strip() + "\n"
    return prompt_txt, chosen_txt, rejected_txt

# process prompt-choosen-rejected pairs
def process_ds(batch, tokenizer, max_len):
    chosen, rejected, prompt_length = [], [], []
    for item in batch:
        # extract prompt and response
        prompt_txt, chosen_txt, rejected_txt = _extract_texts(item)

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
    if _is_hh_dataset(dataset_name) or _is_shp_dataset(dataset_name):
        train_split = config["dataset"].get("subset", "train")
        dataset_id = "Anthropic/hh-rlhf" if _is_hh_dataset(dataset_name) else "stanfordnlp/SHP"
        dataset_dict = load_dataset(dataset_id)

        if isinstance(dataset_dict, dict):
            if train_split in dataset_dict:
                train_raw = dataset_dict[train_split]
            else:
                train_raw = load_dataset(dataset_id, split=train_split)

            if "validation" in dataset_dict:
                val_raw = dataset_dict["validation"]
            elif "test" in dataset_dict:
                val_raw = dataset_dict["test"]
            else:
                split = train_raw.train_test_split(
                    test_size=config["dataset"]["val_ratio"],
                    seed=config["dataset"]["seed"],
                )
                train_raw, val_raw = split["train"], split["test"]
        else:
            split = dataset_dict.train_test_split(
                test_size=config["dataset"]["val_ratio"],
                seed=config["dataset"]["seed"],
            )
            train_raw, val_raw = split["train"], split["test"]

        if _is_hh_dataset(dataset_name):
            train_ds_raw = Dataset.from_list(_build_hh_pairs(train_raw))
            val_ds_raw = Dataset.from_list(_build_hh_pairs(val_raw))
        else:
            train_ds_raw = Dataset.from_list(_build_shp_pairs(train_raw))
            val_ds_raw = Dataset.from_list(_build_shp_pairs(val_raw))
    else:
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



    

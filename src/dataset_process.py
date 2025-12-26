"""Dataset loading and preprocessing utilities for DPO training."""

import torch
from collections import defaultdict
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from functools import partial
from typing import Dict, Iterable, List, Tuple
import tqdm


def extract_anthropic_prompt(prompt_and_response: str) -> str:
    """Extract the Anthropic prompt from a prompt + response string."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    if search_term_idx == -1:
        raise ValueError(f"Prompt does not contain '{search_term}'")
    return prompt_and_response[:search_term_idx + len(search_term)]


def _is_hh_dataset(name: str) -> bool:
    """Return True when the dataset name refers to the HH dataset."""
    name = name.lower()
    return name in {"hh", "hh-rlhf", "anthropic/hh-rlhf"}


def _is_shp_dataset(name: str) -> bool:
    """Return True when the dataset name refers to the SHP dataset."""
    name = name.lower()
    return name in {"shp", "stanfordnlp/shp"}


def _flatten_prompt_pairs(data: Dict[str, Dict[str, List]]) -> List[Dict[str, str]]:
    """Convert prompt-keyed responses/pairs into flat prompt/chosen/rejected rows."""
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


def _build_hh_data(
    dataset: Iterable[Dict[str, str]],
    *,
    silent: bool = True,
) -> Dict[str, Dict[str, List]]:
    """Convert HH rows into a prompt-keyed structure.

    The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.

       For this dataset, the sft_target is just the chosen response.
    """
    data: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc="Processing HH", disable=silent):
        # Extract prompt plus chosen/rejected responses from each row.
        prompt = extract_anthropic_prompt(row["chosen"])
        chosen_response = row["chosen"][len(prompt):]
        rejected_response = row["rejected"][len(prompt):]

        # Track response indices for preference pairs (chosen first, rejected second).
        n_responses = len(data[prompt]["responses"])
        data[prompt]["pairs"].append((n_responses, n_responses + 1))
        data[prompt]["responses"].extend([chosen_response, rejected_response])
        data[prompt]["sft_target"] = chosen_response

    if not silent:
        print(f"Successfully processed {len(data)} prompts.")

    return data


def _build_hh_pairs(dataset: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return HH preference pairs in flat prompt/chosen/rejected format."""
    return _flatten_prompt_pairs(_build_hh_data(dataset))


def _build_shp_data(dataset: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, List]]:
    """Convert SHP rows into prompt-keyed responses/pairs and sft_target."""
    data: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    for row in dataset:
        # SHP stores dialog history separately; build a matching prompt prefix.
        prompt = "\n\nHuman: " + row["history"] + "\n\nAssistant:"
        responses = [" " + row["human_ref_A"], " " + row["human_ref_B"]]
        scores = [row["score_A"], row["score_B"]]

        # Filter out ambiguous preferences based on the score ratio.
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # Translate the SHP label into chosen/rejected ordering.
        n_responses = len(data[prompt]["responses"])
        chosen_first = row["labels"] == 1
        data[prompt]["pairs"].append((n_responses, n_responses + 1) if chosen_first else (n_responses + 1, n_responses))
        data[prompt]["responses"].extend(responses)
        data[prompt]["scores"].extend(scores)

    for prompt, info in data.items():
        scores = info["scores"]
        responses = info["responses"]
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        info["sft_target"] = responses[best_idx]
        del info["scores"]

    return data


def _build_shp_pairs(dataset: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert SHP rows into prompt/chosen/rejected pairs with score filtering."""
    return _flatten_prompt_pairs(_build_shp_data(dataset))


def _extract_texts(item: Dict[str, str]) -> Tuple[str, str, str]:
    """Normalize supported datasets into (prompt, chosen, rejected) text."""
    if "prompt" in item:
        # Already in prompt/chosen/rejected form.
        return item["prompt"], item["chosen"], item["rejected"]

    # Fallback for datasets that use question/chosen/rejected keys.
    prompt_txt = "Prompt: " + item["question"].strip() + "\n"
    chosen_txt = "Response: " + item["chosen"].strip() + "\n"
    rejected_txt = "Response: " + item["rejected"].strip() + "\n"
    return prompt_txt, chosen_txt, rejected_txt


def _build_sft_records(dataset: Iterable[Dict[str, str]], dataset_name: str) -> List[Dict[str, str]]:
    """Build prompt/sft_target records for SFT training."""
    records: List[Dict[str, str]] = []

    if _is_hh_dataset(dataset_name):
        data = _build_hh_data(dataset)
        for prompt, info in data.items():
            records.append(
                {
                    "prompt": prompt,
                    "sft_target": info["sft_target"],
                    "truncation_mode": "keep_end",
                }
            )
        return records

    if _is_shp_dataset(dataset_name):
        data = _build_shp_data(dataset)
        for prompt, info in data.items():
            records.append(
                {
                    "prompt": prompt,
                    "sft_target": info["sft_target"],
                    "truncation_mode": "keep_start",
                }
            )
        return records

    for row in dataset:
        if "prompt" in row and "sft_target" in row:
            prompt_txt = row["prompt"]
            sft_target = row["sft_target"]
        elif "prompt" in row and "chosen" in row:
            prompt_txt = row["prompt"]
            sft_target = row["chosen"]
        else:
            prompt_txt, chosen_txt, _ = _extract_texts(row)
            sft_target = chosen_txt
        records.append(
            {
                "prompt": prompt_txt,
                "sft_target": sft_target,
                "truncation_mode": "keep_start",
            }
        )

    return records


def _truncate_sft_sequence(
    input_ids: List[int],
    labels: List[int],
    max_len: int,
    truncation_mode: str,
) -> Tuple[List[int], List[int]]:
    if len(input_ids) <= max_len:
        return input_ids, labels

    if truncation_mode == "keep_end":
        return input_ids[-max_len:], labels[-max_len:]

    return input_ids[:max_len], labels[:max_len]


def process_sft_ds(batch, tokenizer, max_len):
    """Tokenize SFT examples with prompt masking in the labels, padding, and truncation. 
    Abandoned for now as HF Trainer needs data object not dataloader from build_sft_train_val, 
    logic is implemented in sft_trainer.py"""
    input_ids_list = []
    attention_masks = []
    labels_list = []
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for item in batch:
        prompt = item["prompt"]
        sft_target = item["sft_target"]
        truncation_mode = item.get("truncation_mode", "keep_start")

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(sft_target, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            target_ids = target_ids + [tokenizer.eos_token_id]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        input_ids, labels = _truncate_sft_sequence(
            input_ids,
            labels,
            max_len=max_len,
            truncation_mode=truncation_mode,
        )
        attention_mask = [1] * len(input_ids)

        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
        attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels_list),
    }


def process_ds(batch, tokenizer, max_len):
    """Tokenize a batch of preference pairs for dpo and compute prompt lengths."""
    chosen, rejected, prompt_length = [], [], []
    for item in batch:
        # Extract prompt/chosen/rejected text in a consistent layout.
        prompt_txt, chosen_txt, rejected_txt = _extract_texts(item)

        # Compute prompt length without special tokens for masking later.
        # Tokenizer returns a dict; we only need input_ids.
        prompt_ids = tokenizer(
            prompt_txt,
            add_special_tokens=False,
            return_tensors=None
        )["input_ids"]

        prompt_length.append(len(prompt_ids))

        # Build full prompt+response strings for each side.
        chosen.append(prompt_txt + chosen_txt)
        rejected.append(prompt_txt + rejected_txt)

    # Tokenize with padding and truncation to a fixed max length.
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

    # If BOS is auto-added, shift prompt lengths by one token.
    has_bos = (
        tokenizer.bos_token_id is not None
        and enc_chosen.input_ids.shape[1] > 0
        and enc_chosen.input_ids[0, 0].item() == tokenizer.bos_token_id
    )
    bos_shift = 1 if has_bos else 0

    # Store prompt lengths as a tensor for downstream loss masking.
    prompt_length = torch.tensor(
        [pl + bos_shift for pl in prompt_length], dtype=torch.long
    )

    # Assemble the batch in the shape expected by the trainer.
    out_batch = {
        "prompt_chosen_ids":  enc_chosen.input_ids,          
        "prompt_chosen_mask": enc_chosen.attention_mask,     
        "prompt_rejected_ids":  enc_rejected.input_ids,      
        "prompt_rejected_mask": enc_rejected.attention_mask, 
        "prompt_length": prompt_length,                     
    }

    return out_batch

def build_train_val(config, tokenizer):
    """Create train/val DataLoaders for DPO with dataset-specific preprocessing."""
    dataset_name = config['dataset']['dataset_name']
    if _is_hh_dataset(dataset_name) or _is_shp_dataset(dataset_name):
        train_split = config["dataset"].get("subset", "train")
        dataset_id = "Anthropic/hh-rlhf" if _is_hh_dataset(dataset_name) else "stanfordnlp/SHP"
        dataset_dict = load_dataset(dataset_id)

        # Hugging Face datasets can be a DatasetDict or a single Dataset.
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
                # Create a validation split when the dataset lacks one.
                split = train_raw.train_test_split(
                    test_size=config["dataset"]["val_ratio"],
                    seed=config["dataset"]["seed"],
                )
                train_raw, val_raw = split["train"], split["test"]
        else:
            # Single dataset: split into train/val.
            split = dataset_dict.train_test_split(
                test_size=config["dataset"]["val_ratio"],
                seed=config["dataset"]["seed"],
            )
            train_raw, val_raw = split["train"], split["test"]

        # Convert raw rows into prompt/chosen/rejected training examples.
        if _is_hh_dataset(dataset_name):
            train_ds_raw = Dataset.from_list(_build_hh_pairs(train_raw))
            val_ds_raw = Dataset.from_list(_build_hh_pairs(val_raw))
        else:
            train_ds_raw = Dataset.from_list(_build_shp_pairs(train_raw))
            val_ds_raw = Dataset.from_list(_build_shp_pairs(val_raw))
    else:
        # Generic path for datasets already in prompt/chosen/rejected format.
        ds_raw = load_dataset(dataset_name, split=config['dataset']['subset'])
        if "validation" in ds_raw:
            train_ds_raw = ds_raw["train"]
            val_ds_raw = ds_raw["validation"]
        else:
            # Create a validation split when missing.
            split = ds_raw.train_test_split(test_size=config['dataset']['val_ratio'], seed=config['dataset']['seed'])
            train_ds_raw, val_ds_raw = split["train"], split["test"]

    # Collate function handles tokenization and padding.
    ds_collate = partial(process_ds, tokenizer=tokenizer, max_len=config["dataset"]["max_len"])
    batch_size = config["dpo_training"]["batch_size"]

    # DataLoaders feed tokenized batches to the trainer.
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


def build_sft_train_val(config, tokenizer):
    """Create train/val DataLoaders for SFT using sft_target."""
    dataset_name = config["dataset"]["dataset_name"]
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

        train_records = _build_sft_records(train_raw, dataset_name)
        val_records = _build_sft_records(val_raw, dataset_name)
        train_ds_raw = Dataset.from_list(train_records)
        val_ds_raw = Dataset.from_list(val_records)
    else:
        ds_raw = load_dataset(dataset_name, split=config["dataset"]["subset"])
        if "validation" in ds_raw:
            train_raw = ds_raw["train"]
            val_raw = ds_raw["validation"]
        else:
            split = ds_raw.train_test_split(
                test_size=config["dataset"]["val_ratio"],
                seed=config["dataset"]["seed"],
            )
            train_raw, val_raw = split["train"], split["test"]

        train_ds_raw = Dataset.from_list(_build_sft_records(train_raw, dataset_name))
        val_ds_raw = Dataset.from_list(_build_sft_records(val_raw, dataset_name))

    ds_collate = partial(process_sft_ds, tokenizer=tokenizer, max_len=config["dataset"]["max_len"])
    batch_size = config["sft_training"]["batch_size"]

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



    
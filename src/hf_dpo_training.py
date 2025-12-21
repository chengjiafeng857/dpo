from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
import yaml
import argparse
from model_utils import resolve_torch_dtype

# load yaml config
def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# process_dataset for hugging
def process_ds_hf(item):
    return {
        "prompt":   "Prompt: " + item["question"].strip() + "\n",
        "chosen":   "Response: "   + item["chosen"].strip(),
        "rejected": "Response: "   + item["rejected"].strip(),
    }


# dpo training
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args=parser.parse_args()

    config = load_yaml_config(args.config)

    # load model and tokenizer
    policy_name = config['policy_name']
    torch_dtype = resolve_torch_dtype(config.get("precision"))
    policy = AutoModelForCausalLM.from_pretrained(policy_name, torch_dtype=torch_dtype)
    tok = AutoTokenizer.from_pretrained(policy_name)
    tok.pad_token = tok.eos_token

    ref_name = config['ref_name']
    ref_model = AutoModelForCausalLM.from_pretrained(ref_name, torch_dtype=torch_dtype)

    # freeze ref_model params
    for param in ref_model.parameters():
        param.requires_grad_(False)

    # load dataset
    dataset_name = config['dataset']['dataset_name']
    ds_raw = load_dataset(dataset_name, split=config['dataset']['subset'])
    if "validation" in ds_raw:
        train_ds_raw = ds_raw["train"]
        val_ds_raw = ds_raw["validation"]
    else:
        split = ds_raw.train_test_split(test_size=config['dataset']['val_ratio'], seed=config['dataset']['seed'])
        train_ds_raw, val_ds_raw = split["train"], split["test"]
    train_ds = train_ds_raw.map(process_ds_hf)
    val_ds = val_ds_raw.map(process_ds_hf)

    # precision
    use_bf16 = config['precision'] == 'bf16'

    # dpo config
    dpo_config = DPOConfig(
        learning_rate=float(config['hf_dpo_training']['learning_rate']),
        num_train_epochs=config['hf_dpo_training']['epochs'],
        per_device_train_batch_size=config['hf_dpo_training']['batch_size'],
        beta=config['hf_dpo_training']['beta'],
        save_strategy='epoch',
        eval_strategy='epoch',
        bf16=use_bf16,
        seed=config['hf_dpo_training']['seed'],
        output_dir=config['hf_dpo_training']['save_dir'],
        load_best_model_at_end=True,
        logging_steps=config['hf_dpo_training']['logging_steps'],
        loss_type="sigmoid",
        report_to="wandb"
    )

    # dpo training
    dpo_trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    dpo_trainer.train()
    dpo_trainer.save_model(dpo_config.output_dir)
    tok.save_pretrained(dpo_config.output_dir)

    print(f'Dpo save to {dpo_config.output_dir}')

if __name__ == "__main__":
    train()

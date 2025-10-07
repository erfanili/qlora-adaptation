# train.py (unchanged except for output_dir dynamic via config)
# Usage: accelerate launch --num_processes=4 train.py --config config_lora.yaml
# or --config config_qlora.yaml

import os
from accelerate import PartialState
os.environ["ACCELERATE_DISABLE_WEIGHTS_ONLY_LOADING"] = "true"
os.environ["ACCELERATE_USE_MPS_DEVICE"] = "false"

import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.trainer_utils import set_seed


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def format_dataset(batch, tokenizer, prompt_template, max_length):
    """Format each dialogue-summary pair into input_ids, masks, and labels."""
    input_list, mask_list, label_list = [], [], []
    for dialogue, target in zip(batch["dialogue"], batch["summary"]):
        prompt = prompt_template.format(dialogue=dialogue)
        p = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)
        t = tokenizer(target, add_special_tokens=False, truncation=True, max_length=max_length)

        input_ids = p["input_ids"] + t["input_ids"]
        attention_mask = [1] * len(input_ids)

        # Truncate
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]

        # Labels: mask out prompt tokens
        lbls = [-100] * min(len(p["input_ids"]), max_length) + t["input_ids"]
        lbls = lbls[:max_length]

        # Padding
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            pad_id = tokenizer.eos_token_id
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len
            lbls += [-100] * pad_len

        input_list.append(input_ids)
        mask_list.append(attention_mask)
        label_list.append(lbls)

    return {
        "input_ids": input_list,
        "attention_mask": mask_list,
        "labels": label_list,
    }


def main(args):
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    ds = load_dataset(cfg["dataset"])
    tok = ds.map(
        lambda b: format_dataset(b, tokenizer, cfg["prompt_template"], cfg["max_length"]),
        batched=True,
        remove_columns=ds["train"].column_names,
    )
    train_split = cfg.get("train_split", "train")
    eval_split = cfg.get("eval_split", "validation")

    train_ds = tok[train_split]
    eval_ds = tok[eval_split]

    # Model
    state = PartialState()  # Gets current process/GPU info for DDP
    device_map = {"": state.process_index}  # Load full model on this process's GPU
    if cfg["method"] == "qlora":
        quant_cfg = BitsAndBytesConfig(**cfg["quantization"])
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], quantization_config=quant_cfg, device_map=device_map
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], torch_dtype=torch.float16, device_map=device_map
        )

    model.config.pad_token_id = tokenizer.eos_token_id

    # LoRA
    peft_cfg = LoraConfig(**cfg["peft"])
    model = get_peft_model(model, peft_cfg)

    # Training setup
    train_args_cfg = cfg["training_args"].copy()
    train_args_cfg.setdefault("logging_strategy", "steps")
    train_args_cfg.setdefault("logging_first_step", True)
    train_args_cfg["logging_steps"] = 1  # Log every step for detailed loss monitoring

    ta = TrainingArguments(**train_args_cfg)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args)
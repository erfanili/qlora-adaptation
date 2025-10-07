# eval_three_way.py (updated with dir creation)
# Usage: accelerate launch --num_processes=4 eval_three_way.py --config config.yaml --lora_path ./outputs_lora/checkpoint-XXX --qlora_path ./outputs_qlora/checkpoint-XXX

import os
import yaml
import torch
import json
from datetime import datetime
from accelerate import PartialState, Accelerator
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from peft import PeftModel
import evaluate  # pip install evaluate rouge-score bert-score
import numpy as np
from tqdm import tqdm
from accelerate.utils import gather_object

# Suppress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def format_prompt(dialogue, prompt_template):
    return prompt_template.format(dialogue=dialogue)


def generate_summaries(model, tokenizer, texts, max_new_tokens=128, device_map=None, batch_size=8):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    
    summaries = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        batch_texts = texts[i:i + batch_size]
        outputs = pipe(
            batch_texts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
        batch_summaries = [out[0]["generated_text"].strip() for out in outputs]
        summaries.extend(batch_summaries)
    
    return summaries


def compute_metrics(preds, refs, metrics_list=["rouge", "bertscore"]):
    results = {}
    
    if "rouge" in metrics_list:
        rouge = evaluate.load("rouge")
        results["rouge"] = rouge.compute(predictions=preds, references=refs)
    
    if "bertscore" in metrics_list:
        bertscore = evaluate.load("bertscore")
        bs_results = bertscore.compute(predictions=preds, references=refs, lang="en")
        results["bertscore"] = {
            "precision": np.mean(bs_results["precision"]),
            "recall": np.mean(bs_results["recall"]),
            "f1": np.mean(bs_results["f1"]),
        }
    
    return results


def save_results_to_file(base_metrics, lora_metrics, qlora_metrics, output_dir="./outputs"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"three_way_eval_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    results = {
        "timestamp": timestamp,
        "base_metrics": base_metrics,
        "lora_metrics": lora_metrics,
        "qlora_metrics": qlora_metrics,
        "comparisons": {
            "lora_vs_base": {
                "rouge1_delta": lora_metrics["rouge"]["rouge1"] - base_metrics["rouge"]["rouge1"],
                "rouge2_delta": lora_metrics["rouge"]["rouge2"] - base_metrics["rouge"]["rouge2"],
                "rougeL_delta": lora_metrics["rouge"]["rougeL"] - base_metrics["rouge"]["rougeL"],
                "bertscore_f1_delta": lora_metrics["bertscore"]["f1"] - base_metrics["bertscore"]["f1"],
            },
            "qlora_vs_base": {
                "rouge1_delta": qlora_metrics["rouge"]["rouge1"] - base_metrics["rouge"]["rouge1"],
                "rouge2_delta": qlora_metrics["rouge"]["rouge2"] - base_metrics["rouge"]["rouge2"],
                "rougeL_delta": qlora_metrics["rouge"]["rougeL"] - base_metrics["rouge"]["rougeL"],
                "bertscore_f1_delta": qlora_metrics["bertscore"]["f1"] - base_metrics["bertscore"]["f1"],
            },
            "qlora_vs_lora": {
                "rouge1_delta": qlora_metrics["rouge"]["rouge1"] - lora_metrics["rouge"]["rouge1"],
                "rouge2_delta": qlora_metrics["rouge"]["rouge2"] - lora_metrics["rouge"]["rouge2"],
                "rougeL_delta": qlora_metrics["rouge"]["rougeL"] - lora_metrics["rouge"]["rougeL"],
                "bertscore_f1_delta": qlora_metrics["bertscore"]["f1"] - lora_metrics["bertscore"]["f1"],
            },
        },
    }
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Three-way results saved to: {filepath}")
    return filepath


def main(args):
    accelerator = Accelerator()
    state = PartialState()
    cfg = load_config(args.config)
    
    # Validate paths before proceeding
    if not os.path.exists(args.lora_path):
        raise ValueError(f"LoRA path '{args.lora_path}' does not exist. Ensure training completed and checkpoints saved.")
    if not os.path.exists(args.qlora_path):
        raise ValueError(f"QLoRA path '{args.qlora_path}' does not exist. Ensure training completed and checkpoints saved.")
    if not os.path.exists(os.path.join(args.lora_path, "adapter_config.json")):
        raise ValueError(f"LoRA adapter config missing at '{args.lora_path}'. Check save_strategy in config.")
    if not os.path.exists(os.path.join(args.qlora_path, "adapter_config.json")):
        raise ValueError(f"QLoRA adapter config missing at '{args.qlora_path}'. Check save_strategy in config.")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    ds = load_dataset(cfg["dataset"])
    eval_split = cfg.get("eval_split", "validation")
    eval_ds = ds[eval_split]
    
    # Shard
    eval_ds = eval_ds.shard(state.num_processes, state.process_index)
    dialogues = eval_ds["dialogue"]
    local_refs = eval_ds["summary"]
    prompts = [format_prompt(d, cfg["prompt_template"]) for d in dialogues]
    
    device_map = {"": state.process_index}
    batch_size = cfg["training_args"].get("per_device_eval_batch_size", 8)
    
    # Base model (shared load)
    print(f"[{state.process_index}] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    local_base_summaries = generate_summaries(base_model, tokenizer, prompts, batch_size=batch_size, device_map=device_map)
    accelerator.wait_for_everyone()
    
    all_base_summaries = gather_object(local_base_summaries)
    all_refs = gather_object(local_refs)
    
    # LoRA model
    print(f"[{state.process_index}] Loading LoRA model...")
    lora_model = PeftModel.from_pretrained(base_model, args.lora_path)
    lora_model.eval()
    local_lora_summaries = generate_summaries(lora_model, tokenizer, prompts, batch_size=batch_size, device_map=device_map)
    accelerator.wait_for_everyone()
    all_lora_summaries = gather_object(local_lora_summaries)
    
    # QLoRA model
    print(f"[{state.process_index}] Loading QLoRA model...")
    quant_cfg = BitsAndBytesConfig(**cfg["quantization"])
    qlora_base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quant_cfg,
        torch_dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    qlora_model = PeftModel.from_pretrained(qlora_base, args.qlora_path)
    qlora_model.eval()
    local_qlora_summaries = generate_summaries(qlora_model, tokenizer, prompts, batch_size=batch_size, device_map=device_map)
    accelerator.wait_for_everyone()
    all_qlora_summaries = gather_object(local_qlora_summaries)
    
    if state.process_index == 0:
        base_metrics = compute_metrics(all_base_summaries, all_refs)
        lora_metrics = compute_metrics(all_lora_summaries, all_refs)
        qlora_metrics = compute_metrics(all_qlora_summaries, all_refs)
        
        print("Base metrics:", base_metrics)
        print("LoRA metrics:", lora_metrics)
        print("QLoRA metrics:", qlora_metrics)
        
        # Comparisons
        print("\nLoRA vs Base:")
        for k, v in lora_metrics["rouge"].items():
            print(f"{k}: +{v - base_metrics['rouge'][k]:.4f}")
        print(f"BERTScore F1: +{lora_metrics['bertscore']['f1'] - base_metrics['bertscore']['f1']:.4f}")
        
        print("\nQLoRA vs Base:")
        for k, v in qlora_metrics["rouge"].items():
            print(f"{k}: +{v - base_metrics['rouge'][k]:.4f}")
        print(f"BERTScore F1: +{qlora_metrics['bertscore']['f1'] - base_metrics['bertscore']['f1']:.4f}")
        
        print("\nQLoRA vs LoRA:")
        for k, v in qlora_metrics["rouge"].items():
            print(f"{k}: +{v - lora_metrics['rouge'][k]:.4f}")
        print(f"BERTScore F1: +{qlora_metrics['bertscore']['f1'] - lora_metrics['bertscore']['f1']:.4f}")
        
        save_results_to_file(base_metrics, lora_metrics, qlora_metrics)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lora_path", required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--qlora_path", required=True, help="Path to QLoRA checkpoint")
    args = parser.parse_args()
    main(args)
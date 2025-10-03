import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.trainer_utils import set_seed
import os, json,time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = "meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset",default = "knkarthick/samsum")
    parser.add_argument("--method",default = "qlora")
    parser.add_argument("--output_dir", default = "./outputs")
    parser.add_argument("--seed",default = 42)
    
    return parser.parse_args()


def format_dataset(batch,tokenizer,max_length = 512):
    inputs = [f"{dialogue}\n\nSummary: {summary}" for dialogue,summary in zip(batch["dialogue"],batch["summary"])]
    model_inputs = tokenizer(inputs, max_length = max_length, truncation = True, padding = "max_length")

    return model_inputs



def main():
    args = parse_args()
    seed = set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name,use_fast = True)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset(args.dataset)
    tokenized = dataset.map(lambda x:format_dataset(x, tokenizer), batched=True)

    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]
    
    device_map=  {"": torch.cuda.current_device()}
    if args.method == "qlora":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_4bit = True,
            device_map = device_map,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_use_double_quant = True,
            bnb_4bit_compute_dtype = torch.float16
        )
        model = prepare_model_for_kbit_training(model)
    
    
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype= torch.float16,
            device_map = device_map
        )
    peft_config = LoraConfig(
        r = 16,
        lora_alpha = 32,
        lora_dropout = 0.05,
        target_modules = ["q_proj", "v_proj"],
        bias = "none",
        task_type = "CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    run_name = f"{args.method}-samsum"
    training_args = TrainingArguments(
        output_dir = os.path.join(args.output_dir,run_name),
        overwrite_output_dir = True,
        num_train_epochs = 1,
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,
        gradient_accumulation_steps = 8,
        learning_rate = 2e-4,
        fp16 = True,
        eval_strategy = "epoch",
        logging_dir = "./logs",
        logging_steps = 5,
        report_to = "none",
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer,mlm = False)
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = data_collator,
    )
    
    start_mem = torch.cuda.memory_allocated()/1e9
    start_time = time.time()
    
    trainer.train()
    
    end_time = time.time()
    end_mem = torch.cuda.max_memory_allocated()/1e9
    
    trainer.save_model()
    
    
    stats = {
        "method": args.method,
        "train_time_hours": round((end_time-start_time)/3600,2),
        "gpu_mem_gb": round(end_mem,2)
    }
    os.makedirs("results",exist_ok = True)
    with open(f"results/{args.method}_stats.json",'w') as f:
        json.dump(stats,f,indent=2)
        
    print("==== Training finished ====")
    print(stats)
    
    
if __name__ == "__main__":
    main()
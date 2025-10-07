# QLoRA Adaptation for Dialogue Summarization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-lightblue)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository demonstrates efficient fine-tuning of the Llama-3.2-1B model for dialogue summarization on the [SAMSum dataset](https://huggingface.co/datasets/knkarthick/samsum) using LoRA and QLoRA (quantized LoRA). It includes training scripts, multi-GPU support via Accelerate, and evaluation with ROUGE and BERTScore metrics.

The pipeline trains both methods sequentially, evaluates them three-way against the base model, and saves logs/metrics systematically (TensorBoard for visualization).

## Features
- **Efficient Fine-Tuning**: LoRA (full fp16) and QLoRA (4-bit NF4) for low-rank adaptation.
- **Multi-GPU Training**: Distributed via Accelerate (DDP).
- **Custom Dataset Formatting**: Prompt-based seq2seq for summarization.
- **Metrics**: ROUGE (1/2/L) and BERTScore (F1) on validation set.
- **Systematic Logging**: TensorBoard for loss/LR curves.

## Installation
1. Clone the repo:
   ```bash
   git clone <your-repo-url>
   cd qlora-adaptation
   pip install -r requirements.txt

##Usage

1. Update config.yaml (base template—method/output_dir overridden per run).
2. Run the full pipeline:

```
sh run_pipeline.sh --num_gpus 2
```
Step 0: Base model eval (placeholder, generates summaries from zero-shot base).
Step 1: Train LoRA (`config_lora.yaml`).
Step 2: Train QLoRA (`config_qlora.yaml`)
Step 3: Three-way eval (base vs. LoRA vs. QLoRA)


View logs: `tensorboard --logdir ./logs_lora` (or `./logs_qlora`).
Manual train: `accelerate launch --num_processes=4 train.py --config config_lora.yaml.`
Standalone eval: `accelerate launch --num_processes=4 eval_three_way.py --config config.yaml --lora_path ./outputs_lora/checkpoint-last --qlora_path ./outputs_qlora/checkpoint-last`.

## Results
Evaluated on SAMSum validation set (~1,450 samples). LoRA shows superior gains over base; QLoRA is efficient but trails slightly due to quantization noise on this 1B model.

| Model   | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---------|---------|---------|---------|--------------|
| **Base** | 0.1788 | 0.0595 | 0.1443 | 0.8468      |
| **LoRA** | 0.3508 | 0.1742 | 0.2858 | 0.8871      |
| **QLoRA**| 0.2868 | 0.1382 | 0.2345 | 0.8729      |


| Comparison     | ROUGE-1 Δ | ROUGE-2 Δ | ROUGE-L Δ | BERTScore F1 Δ |
|----------------|-----------|-----------|-----------|----------------|
| **LoRA vs. Base** | +0.1720 | +0.1147 | +0.1415 | +0.0402       |
| **QLoRA vs. Base** | +0.1080 | +0.0787 | +0.0902 | +0.0261       |
| **QLoRA vs. LoRA** | -0.0640 | -0.0360 | -0.0513 | -0.0142       |


Insights: LoRA achieves ~17% ROUGE-1 improvement (best for precision); QLoRA saves ~50% VRAM with ~11% gains. Full dataset run (Oct 07, 2025).
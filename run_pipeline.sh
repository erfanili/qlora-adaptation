#!/bin/bash
# sequential_train_eval.sh
# Usage: ./sequential_train_eval.sh [--num_gpus 4] [--config_base config.yaml]
# Trains LoRA then QLoRA sequentially, then runs three-way eval on best checkpoints.
# Assumes: train.py, eval_three_way.py, config_lora.yaml, config_qlora.yaml in current dir.
# Best checkpoint: Assumes last checkpoint (adjust if using early stopping).

set -e  # Exit on error

NUM_GPUS=${1:-3}
CONFIG_BASE=${2:-config.yaml}
LO_CONFIG="config_lora.yaml"
QL_CONFIG="config_qlora.yaml"

# Create method-specific configs if not exist
if [ ! -f "$LO_CONFIG" ]; then
    echo "Creating $LO_CONFIG from $CONFIG_BASE..."
    sed "s/output_dir: .*/output_dir: \"./outputs_lora\"/" "$CONFIG_BASE" > "$LO_CONFIG"
    sed -i "s/method: .*/method: \"lora\"/" "$LO_CONFIG"
    sed -i "s/logging_dir: .*/logging_dir: \"./logs_lora\"/" "$LO_CONFIG"
fi

if [ ! -f "$QL_CONFIG" ]; then
    echo "Creating $QL_CONFIG from $CONFIG_BASE..."
    sed "s/output_dir: .*/output_dir: \"./outputs_qlora\"/" "$CONFIG_BASE" > "$QL_CONFIG"
    sed -i "s/method: .*/method: \"qlora\"/" "$QL_CONFIG"
    sed -i "s/logging_dir: .*/logging_dir: \"./logs_qlora\"/" "$QL_CONFIG"
fi

# Step 1: Train LoRA
echo "=== Training LoRA ==="
accelerate launch --num_processes=$NUM_GPUS train.py --config "$LO_CONFIG"

# Step 2: Train QLoRA
echo "=== Training QLoRA ==="
accelerate launch --num_processes=$NUM_GPUS train.py --config "$QL_CONFIG"

# Step 3: Find best checkpoints (last one for now; update for eval_loss min)
LO_PATH=$(ls -td ./outputs_lora/checkpoint-* 2>/dev/null | head -1)
QL_PATH=$(ls -td ./outputs_qlora/checkpoint-* 2>/dev/null | head -1)

if [ -z "$LO_PATH" ] || [ ! -d "$LO_PATH" ] || [ ! -f "$LO_PATH/adapter_config.json" ]; then
    echo "Error: LoRA checkpoint invalid/missing (no checkpoints in ./outputs_lora/). Check training logs and save_strategy."
    exit 1
fi

if [ -z "$QL_PATH" ] || [ ! -d "$QL_PATH" ] || [ ! -f "$QL_PATH/adapter_config.json" ]; then
    echo "Error: QLoRA checkpoint invalid/missing (no checkpoints in ./outputs_qlora/). Check training logs and save_strategy."
    exit 1
fi

echo "Using LoRA checkpoint: $LO_PATH"
echo "Using QLoRA checkpoint: $QL_PATH"

# Step 4: Three-way eval
echo "=== Three-way Evaluation ==="
accelerate launch --num_processes=$NUM_GPUS eval_three_way.py --config "$CONFIG_BASE" --lora_path "$LO_PATH" --qlora_path "$QL_PATH"

echo "Sequential pipeline complete! Check outputs_lora/, outputs_qlora/, logs_* for results."
echo "View logs: tensorboard --logdir ./logs_lora  # or ./logs_qlora"
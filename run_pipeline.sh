#!/bin/bash
set -e

# Configuration
SFT_OUTPUT_DIR="models/sft_final"
DPO_OUTPUT_DIR="models/dpo_final"
PYTHON=".venv/bin/python3"

echo "==========================================="
echo "   Starting SFT -> DPO Training Pipeline   "
echo "==========================================="

# 1. Setup Environment (if not exists)
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    $PYTHON -m pip install torch transformers wandb datasets uv
fi

# Ensure dependencies are installed (fast check)
# $PYTHON -m pip install -q torch transformers wandb datasets

# 2. Run SFT
echo ""
echo ">>> [Stage 1] Starting Supervised Fine-Tuning (SFT)..."
echo "    Output Directory: $SFT_OUTPUT_DIR"
# Running with small offset/epochs for demonstration speed
WANDB_MODE=offline $PYTHON dpo/src/sft_training.py \
    --model_name "gpt2" \
    --output_dir "$SFT_OUTPUT_DIR" \
    --epochs 1 \
    --offset 500 \
    --batch_size 2 \
    --wandb_project "sft-pipeline-demo"

echo ">>> [Stage 1] SFT Complete."

# 3. Run DPO (Loading SFT Model)
echo ""
echo ">>> [Stage 2] Starting Direct Preference Optimization (DPO)..."
echo "    Loading Model From: $SFT_OUTPUT_DIR"
echo "    Output Directory: $DPO_OUTPUT_DIR"

WANDB_MODE=offline $PYTHON dpo/src/dpo_training.py \
    --model_name "$SFT_OUTPUT_DIR" \
    --output_dir "$DPO_OUTPUT_DIR" \
    --epochs 1 \
    --offset 500 \
    --beta 0.1 \
    --batch_size 2 \
    --wandb_project "dpo-pipeline-demo"

echo ""
echo "==========================================="
echo "       Pipeline Execution Completed        "
echo "==========================================="
echo "Final DPO Model saved to: $DPO_OUTPUT_DIR"

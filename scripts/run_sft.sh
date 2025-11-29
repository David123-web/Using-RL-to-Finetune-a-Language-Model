#!/bin/bash

# Script to run supervised fine-tuning

echo "================================"
echo "Running Supervised Fine-Tuning"
echo "================================"

# Default config path
CONFIG_PATH="${1:-config/sft_config.yaml}"

echo "Using config: $CONFIG_PATH"
echo ""

# Run training
python -m src.training.train_sft --config "$CONFIG_PATH"

echo ""
echo "================================"
echo "SFT Training Complete!"
echo "================================"

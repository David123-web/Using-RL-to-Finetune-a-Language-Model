#!/bin/bash

# Script to run PPO training

echo "================================"
echo "Running PPO Training"
echo "================================"

# Default config paths
PPO_CONFIG="${1:-config/ppo_config.yaml}"
REWARD_CONFIG="${2:-config/reward_config.yaml}"

echo "Using PPO config: $PPO_CONFIG"
echo "Using reward config: $REWARD_CONFIG"
echo ""

# Run training
python -m src.training.train_ppo --config "$PPO_CONFIG" --reward_config "$REWARD_CONFIG"

echo ""
echo "================================"
echo "PPO Training Complete!"
echo "================================"

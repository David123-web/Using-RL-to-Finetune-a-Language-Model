#!/bin/bash

# Script to run PPO training

echo "================================"
echo "Running PPO Training"
echo "================================"

# Default config paths
PPO_CONFIG="config/ppo_config.yaml"
REWARD_CONFIG="config/reward_config.yaml"

if [ $# -ge 1 ]; then
	PPO_CONFIG="$1"
	shift
fi

if [ $# -ge 1 ]; then
	REWARD_CONFIG="$1"
	shift
fi

EXTRA_ARGS=("$@")

echo "Using PPO config: $PPO_CONFIG"
echo "Using reward config: $REWARD_CONFIG"
echo ""

# Run training
python -m src.training.train_ppo --config "$PPO_CONFIG" --reward_config "$REWARD_CONFIG" "${EXTRA_ARGS[@]}"

echo ""
echo "================================"
echo "PPO Training Complete!"
echo "================================"

#!/bin/bash

# Run PPO training across multiple random seeds.

set -e

PPO_CONFIG="config/ppo_config.yaml"
REWARD_CONFIG="config/reward_config.yaml"
SEEDS="11,22,33"
SAVE_ROOT="models/policy_ppo_multiseed"

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      PPO_CONFIG="$2"
      shift 2
      ;;
    --reward_config)
      REWARD_CONFIG="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --save_root)
      SAVE_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--config PATH] [--reward_config PATH] [--seeds 11,22,33] [--save_root DIR]"
      exit 1
      ;;
  esac
done

echo "================================"
echo "Running Multi-Seed PPO Training"
echo "================================"
echo "PPO config: $PPO_CONFIG"
echo "Reward config: $REWARD_CONFIG"
echo "Seeds: $SEEDS"
echo "Save root: $SAVE_ROOT"
echo ""

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"

for SEED in "${SEED_LIST[@]}"; do
  OUT_DIR="$SAVE_ROOT/seed_${SEED}"
  echo "--------------------------------"
  echo "Training seed $SEED"
  echo "Output dir: $OUT_DIR"
  echo "--------------------------------"

  python -m src.training.train_ppo \
    --config "$PPO_CONFIG" \
    --reward_config "$REWARD_CONFIG" \
    --seed "$SEED" \
    --save_dir "$OUT_DIR"

  echo "Completed seed $SEED"
  echo ""
done

echo "================================"
echo "Multi-Seed PPO Training Complete"
echo "================================"

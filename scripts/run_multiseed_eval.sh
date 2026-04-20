#!/bin/bash

# Run evaluation with multiple evaluation seeds and aggregate the results.

set -e

CONFIG_PATH="config/model_config.yaml"
REWARD_CONFIG_PATH="config/reward_config.yaml"
SEEDS="42,43,44"
MODES="greedy,sampling"
OUTPUT_ROOT="results/multiseed_eval"
BASE=false
SFT_PATH=""
PPO_PATH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --reward_config)
      REWARD_CONFIG_PATH="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --modes)
      MODES="$2"
      shift 2
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --base)
      BASE=true
      shift
      ;;
    --sft)
      SFT_PATH="$2"
      shift 2
      ;;
    --ppo)
      PPO_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--base] [--sft PATH] [--ppo PATH] [--seeds 42,43,44] [--modes greedy,sampling] [--output_root DIR]"
      exit 1
      ;;
  esac
done

echo "================================"
echo "Running Multi-Seed Evaluation"
echo "================================"
echo "Model config: $CONFIG_PATH"
echo "Reward config: $REWARD_CONFIG_PATH"
echo "Seeds: $SEEDS"
echo "Modes: $MODES"
echo "Output root: $OUTPUT_ROOT"
echo ""

mkdir -p "$OUTPUT_ROOT"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"

for SEED in "${SEED_LIST[@]}"; do
  OUT_DIR="$OUTPUT_ROOT/seed_${SEED}"
  mkdir -p "$OUT_DIR"

  CMD=(python -m src.evaluation.evaluate --output "$OUT_DIR" --config "$CONFIG_PATH" --reward_config "$REWARD_CONFIG_PATH" --seed "$SEED" --modes "$MODES")

  if [ "$BASE" = true ]; then
    CMD+=(--base)
  fi

  if [ -n "$SFT_PATH" ]; then
    CMD+=(--sft "$SFT_PATH")
  fi

  if [ -n "$PPO_PATH" ]; then
    CMD+=(--ppo "$PPO_PATH")
  fi

  echo "--------------------------------"
  echo "Evaluating seed $SEED"
  echo "Command: ${CMD[*]}"
  echo "--------------------------------"

  "${CMD[@]}"
  echo ""
done

AGG_OUTPUT="$OUTPUT_ROOT/aggregate.json"
python -m src.evaluation.aggregate_multiseed --input_root "$OUTPUT_ROOT" --output "$AGG_OUTPUT"

echo "================================"
echo "Multi-Seed Evaluation Complete"
echo "Aggregate file: $AGG_OUTPUT"
echo "================================"

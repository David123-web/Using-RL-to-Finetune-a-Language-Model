#!/bin/bash

# Script to run evaluation

echo "================================"
echo "Running Model Evaluation"
echo "================================"

# Parse arguments
BASE=false
SFT_PATH=""
PPO_PATH=""
OUTPUT_DIR="results"
SEED=""
MODES=""

while [[ $# -gt 0 ]]; do
  case $1 in
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
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --modes)
      MODES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--base] [--sft PATH] [--ppo PATH] [--output DIR] [--seed N] [--modes greedy,sampling]"
      exit 1
      ;;
  esac
done

# Build command
CMD=(python -m src.evaluation.evaluate --output "$OUTPUT_DIR")

if [ "$BASE" = true ]; then
  CMD+=(--base)
fi

if [ -n "$SFT_PATH" ]; then
  CMD+=(--sft "$SFT_PATH")
fi

if [ -n "$PPO_PATH" ]; then
  CMD+=(--ppo "$PPO_PATH")
fi

if [ -n "$SEED" ]; then
  CMD+=(--seed "$SEED")
fi

if [ -n "$MODES" ]; then
  CMD+=(--modes "$MODES")
fi

echo "Running: ${CMD[*]}"
echo ""

# Run evaluation
"${CMD[@]}"

echo ""
echo "================================"
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "================================"

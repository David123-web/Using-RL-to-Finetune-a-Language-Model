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
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--base] [--sft PATH] [--ppo PATH] [--output DIR]"
      exit 1
      ;;
  esac
done

# Build command
CMD="python -m src.evaluation.evaluate --output $OUTPUT_DIR"

if [ "$BASE" = true ]; then
  CMD="$CMD --base"
fi

if [ -n "$SFT_PATH" ]; then
  CMD="$CMD --sft $SFT_PATH"
fi

if [ -n "$PPO_PATH" ]; then
  CMD="$CMD --ppo $PPO_PATH"
fi

echo "Running: $CMD"
echo ""

# Run evaluation
eval $CMD

echo ""
echo "================================"
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "================================"

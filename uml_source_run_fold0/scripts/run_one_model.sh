#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

model="${1:?Usage: bash scripts/run_one_model.sh llama|qwen}"

case "$model" in
  llama|qwen) ;;
  *) echo "Model must be llama or qwen"; exit 1 ;;
esac

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

run_root="runs/within/fold0/${model}/seed42"
result_root="results/within/fold0/${model}/seed42"

echo "STEP 1: Evaluate ${model} base model"

python -u src/evaluate.py \
  --model "$model" \
  --fold 0 \
  --seed 42 \
  --views original \
  --rank \
  --out "${result_root}/baseline"

echo "STEP 2: Train ${model} SFT adapter"

python -u src/train.py \
  --model "$model" \
  --method sft \
  --fold 0 \
  --seed 42 \
  --out "${run_root}/sft"

echo "STEP 3: Evaluate ${model} SFT adapter"

python -u src/evaluate.py \
  --model "$model" \
  --fold 0 \
  --seed 42 \
  --adapter "${run_root}/sft/best" \
  --views original \
  --rank \
  --out "${result_root}/sft"

echo "STEP 4: Train ${model} contrastive adapter"

python -u src/train.py \
  --model "$model" \
  --method contrastive \
  --fold 0 \
  --seed 42 \
  --out "${run_root}/contrastive"

echo "STEP 5: Evaluate ${model} contrastive adapter"

python -u src/evaluate.py \
  --model "$model" \
  --fold 0 \
  --seed 42 \
  --adapter "${run_root}/contrastive/best" \
  --views original \
  --rank \
  --out "${result_root}/contrastive"

echo "COMPLETED: ${model}"

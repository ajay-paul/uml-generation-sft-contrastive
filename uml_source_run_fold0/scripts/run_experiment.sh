#!/usr/bin/env bash
set -euo pipefail
fold="${1:-0}"
seed="${2:-42}"
export TOKENIZERS_PARALLELISM=false
for model in llama qwen; do
  root="runs/within/fold${fold}/${model}/seed${seed}"
  result="results/within/fold${fold}/${model}/seed${seed}"
  python src/evaluate.py --model "$model" --fold "$fold" --seed "$seed" --out "${result}/baseline" --views original --rank
  for method in sft contrastive; do
    python src/train.py --model "$model" --method "$method" --fold "$fold" --seed "$seed" --out "${root}/${method}"
    python src/evaluate.py --model "$model" --fold "$fold" --seed "$seed" --adapter "${root}/${method}/best" --out "${result}/${method}" --views original --rank
  done
done
python src/report_results.py --root results --fold "$fold" --seed "$seed" --out "results/report_fold${fold}_seed${seed}"

#!/usr/bin/env bash
# Run from the prepared uml_source_run_fold0 project, with its venv active.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -c 'import matplotlib' || { echo 'Install plotting dependency: python -m pip install matplotlib'; exit 1; }
fold="${1:-0}"
seed="${2:-42}"
out="${3:-results_test10_$(date +%Y%m%d_%H%M%S)}"
for model in llama qwen; do
  for method in sft contrastive; do
    adapter="runs/within/fold${fold}/${model}/seed${seed}/${method}/best"
    test -f "$adapter/adapter_config.json" || { echo "Missing adapter: $adapter"; exit 1; }
  done
done
python src/test_10_requirements.py prepare --out "$out" --fold "$fold" --seed "$seed"
for model in llama qwen; do
  for method in baseline sft contrastive; do
    python -u src/test_10_requirements.py run --out "$out" --fold "$fold" --seed "$seed" \
      --model "$model" --condition "$method" 2>&1 | tee "$out/${model}_${method}.log"
  done
done
python src/test_10_requirements.py summarize --out "$out" --fold "$fold" --seed "$seed"
echo "Finished. Results: $out/comparison.csv"

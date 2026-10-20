#!/usr/bin/env bash
# Steps 3-5: train every model on N different splits, then compare.
# Usage: bash run_models.sh [N_SPLITS]   (default 5)
set -e
N=${1:-5}
cd "$(dirname "$0")/src"
rm -rf ../results/runs
for s in $(seq 0 $((N - 1))); do
  echo "=================== split $s ==================="
  python train_xgb.py --variant tabular --gpu --split-seed $s
  python train_xgb.py --variant links --gpu --split-seed $s
  python train_gnn.py --no-graph --split-seed $s
  python train_gnn.py --split-seed $s
  python train_gnn.py --stack --split-seed $s
  python train_xgb.py --variant hybrid --gpu --split-seed $s
done
python compare.py

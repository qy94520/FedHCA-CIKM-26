#!/usr/bin/env bash
set -euo pipefail

python src/fl_main.py \
  --dataset cifar100 \
  --num_clients 10 \
  --epochs_local 10 \
  --task_global_round 10 \
  --baseclass 10 \
  --learnedclasses 10 \
  --incre_tasks 10 \
  --method FedHCA \
  --output_root outputs

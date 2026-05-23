#!/usr/bin/env bash
set -euo pipefail

python src/fl_main.py \
  --dataset cifar100 \
  --num_users 10 \
  --epochs 10 \
  --local_ep 10 \
  --baseclass 10 \
  --increclass 10 \
  --method FedHCCA \
  --output_root outputs

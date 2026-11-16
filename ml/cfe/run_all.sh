#!/usr/bin/env bash
set -euo pipefail

M=Qwen/Qwen3.5-0.8B
A=runs/cfe-0.8b/adapter

python cfe/prepare.py

python sft/train.py --model $M --train data/cfe_train.jsonl --out runs/cfe-0.8b \
  --epochs 2 --batch 2 --grad-accum 8 --max-len 6144 --lr 2e-4

python cfe/predict.py --model $M --adapter $A --calls data/raw/val.jsonl --out runs/cfe-0.8b/val
python cfe/predict.py --model $M --adapter $A --calls data/val_adv.jsonl --out runs/cfe-0.8b/val_adv
python cfe/predict.py --model $M --adapter $A --calls data/raw/val.jsonl --batch 1 --out runs/cfe-0.8b/val_b1
python cfe/predict.py --model $M --calls data/raw/val.jsonl --out runs/zeroshot-0.8b/val
python cfe/predict.py --model $M --adapter $A --calls data/raw/val.jsonl --no-close --out runs/cfe-0.8b/val_noclose
python cfe/predict.py --model $M --adapter $A --calls data/raw/val.jsonl --no-snap --out runs/cfe-0.8b/val_nosnap

python cfe/report.py --gold data/raw/val.jsonl \
  --runs runs/zeroshot-0.8b/val runs/cfe-0.8b/val runs/cfe-0.8b/val_b1 runs/cfe-0.8b/val_noclose runs/cfe-0.8b/val_nosnap \
  --out runs/summary_val.md
python cfe/report.py --gold data/val_adv.jsonl --runs runs/cfe-0.8b/val_adv --out runs/summary_adv.md

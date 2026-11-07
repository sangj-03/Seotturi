#!/usr/bin/env bash
# Stage 1: train on Gangwon (118) first, while the other regions finish downloading.
# Each step is skipped when its output exists; training resumes from checkpoints.
set -euo pipefail
cd "$(dirname "$0")"
PY=/data/seotturi/.venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
step() { echo "$(date '+%F %T') == $*"; }

if [ ! -s data_118/train.jsonl ]; then
    step "prepare Gangwon data"
    $PY prepare_data.py --datasets 118 --train-per-region 25000 --valid-per-region 300 --out data_118
fi
if [ ! -s runs/base_eval_118/summary.json ]; then
    step "evaluate base model on Gangwon"
    $PY evaluate.py --manifest data_118/valid.jsonl --out runs/base_eval_118
fi
if [ ! -d runs/lora_gangwon/final ]; then
    step "train Gangwon adapter"
    resume=()
    ls -d runs/lora_gangwon/checkpoint-* >/dev/null 2>&1 && resume=(--resume)
    $PY train.py --train data_118/train.jsonl --valid data_118/valid.jsonl --out runs/lora_gangwon --epochs 1 "${resume[@]}"
fi
if [ ! -s runs/lora_gangwon_eval/summary.json ]; then
    step "evaluate Gangwon adapter"
    $PY evaluate.py --manifest data_118/valid.jsonl --adapter runs/lora_gangwon/final --out runs/lora_gangwon_eval
fi
step "stage 1 done"

#!/bin/bash
# The 1,199-utterance validation set (data_all/valid.jsonl, about 300 per region) for v1 and the stage-4
# checkpoints, the same script and decoding for all: the engine's 160-clip benchmark is too small to tell
# 0.01 apart. Killed first if memory runs out (the site engine already has oom_score_adj 1000).
#   setsid nohup ./run_eval_full.sh > runs/eval_full/run.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
echo 1000 > /proc/self/oom_score_adj
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
REPO=$(cd ../.. && pwd -P)   # repository root
PY=${PY:-$REPO/.venv/bin/python}
for spec in v1:runs/lora_v1/final v3_500:runs/lora_v3/checkpoint-500 v3_750:runs/lora_v3/checkpoint-750 v3_1000:runs/lora_v3/checkpoint-1000; do
  name=${spec%%:*}; adapter=${spec#*:}
  [ -f runs/eval_full/$name/summary.json ] && { echo "== $name done already"; continue; }
  echo "== $(date '+%F %T') $name ($adapter)"
  $PY evaluate.py --manifest data_all/valid.jsonl --adapter "$adapter" --out runs/eval_full/$name
done
echo "== $(date '+%F %T') all done"

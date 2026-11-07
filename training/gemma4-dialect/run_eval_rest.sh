#!/bin/bash
# After the v1 evaluation (run_eval_full.sh, stopped after v1): the stage-4 checkpoints on ONE loaded base
# model (evaluate.py --adapters), then score_eval.py's report. Waits until 45 GB are free before loading:
# loading the 12B model took the shared memory from 54 GB to 3 GB for a moment beside the site's engine.
#   setsid nohup ./run_eval_rest.sh >> runs/eval_full/run.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
echo 1000 > /proc/self/oom_score_adj
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
REPO=$(cd ../.. && pwd -P)   # repository root
PY=${PY:-$REPO/.venv/bin/python}
while pgrep -f 'evaluate.py --manifest data_all/valid.jsonl --adapter runs/lora_v1/final' > /dev/null; do sleep 60; done
specs="v3_500=runs/lora_v3/checkpoint-500,v3_750=runs/lora_v3/checkpoint-750,v3_1000=runs/lora_v3/checkpoint-1000"
[ -f runs/eval_full/v1/summary.json ] || specs="v1=runs/lora_v1/final,$specs"
for i in $(seq 1 180); do
  avail=$(free -g | awk 'NR==2{print $7}')
  [ "$avail" -ge 45 ] && break
  [ $((i % 10)) -eq 1 ] && echo "$(date '+%F %T') waiting for memory: ${avail} GB free, need 45"
  sleep 60
done
avail=$(free -g | awk 'NR==2{print $7}')
if [ "$avail" -lt 45 ]; then echo "$(date '+%F %T') gave up: only ${avail} GB free after 3 hours"; exit 1; fi
echo "== $(date '+%F %T') adapters $specs (${avail} GB free)"
$PY evaluate.py --manifest data_all/valid.jsonl --adapters "$specs" --out runs/eval_full
echo "== $(date '+%F %T') scoring"
$PY score_eval.py runs/eval_full/v1 runs/eval_full/v3_500 runs/eval_full/v3_750 runs/eval_full/v3_1000 --base v1 \
  --out runs/eval_full/report.json > runs/eval_full/report.txt 2>&1
echo "== $(date '+%F %T') done (runs/eval_full/report.txt)"

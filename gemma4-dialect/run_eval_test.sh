#!/bin/bash
# Independent test set: 1,000 utterances per region from AI Hub 2.Validation recordings that neither
# data_all/valid.jsonl (training validation, checkpoint choice) nor the site benchmark touched, at most 4 per
# recording. Then v1 and the stage-4 checkpoints 500 / 1000 on ONE loaded base model, and score_eval.py's report.
# This set confirms results; checkpoints are still chosen on data_all/valid.jsonl.
#   setsid nohup ./run_eval_test.sh >> runs/eval_test/run.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
echo 1000 > /proc/self/oom_score_adj
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
PY=/data/seotturi/.venv/bin/python
if [ ! -s data_test/test.jsonl ]; then
  echo "== $(date '+%F %T') prepare data_test"
  $PY prepare_data.py --datasets 118,119,121,122 --train-per-region 0 --valid-per-region 1000 \
    --exclude-recordings data_all/valid.jsonl,/data/seotturi/seotturi-web/state/benchmark.jsonl \
    --max-per-recording 4 --seed 2 --out data_test || exit 1
  mv data_test/valid.jsonl data_test/test.jsonl || exit 1
fi
# loading the 12B model takes the shared memory down by ~45 GB for a moment beside the site's engine
for i in $(seq 1 180); do
  avail=$(free -g | awk 'NR==2{print $7}')
  [ "$avail" -ge 45 ] && break
  [ $((i % 10)) -eq 1 ] && echo "$(date '+%F %T') waiting for memory: ${avail} GB free, need 45"
  sleep 60
done
avail=$(free -g | awk 'NR==2{print $7}')
if [ "$avail" -lt 45 ]; then echo "$(date '+%F %T') gave up: only ${avail} GB free after 3 hours"; exit 1; fi
specs="v1=runs/lora_v1/final,v3_500=runs/lora_v3/checkpoint-500,v3_1000=runs/lora_v3/checkpoint-1000"
echo "== $(date '+%F %T') adapters $specs (${avail} GB free)"
$PY evaluate.py --manifest data_test/test.jsonl --per-region 1000 --adapters "$specs" --out runs/eval_test || exit 1
echo "== $(date '+%F %T') scoring"
$PY score_eval.py runs/eval_test/v1 runs/eval_test/v3_500 runs/eval_test/v3_1000 --base v1 \
  --out runs/eval_test/report.json > runs/eval_test/report.txt 2>&1
echo "== $(date '+%F %T') done (runs/eval_test/report.txt)"

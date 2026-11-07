#!/bin/bash
# lora_v1 and the served lora_r64c on data_test, decoded as the web engine serves (evaluate.py --guard: forced
# "방언:" start, loop stop and repair), then score_eval.py's report (runs/eval_test_guard/report.txt).
#   setsid nohup ./run_eval_guard.sh >> runs/eval_test_guard/run.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
echo 1000 > /proc/self/oom_score_adj
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
REPO=$(cd ../.. && pwd -P)   # repository root
PY=${PY:-$REPO/.venv/bin/python}
# loading the 12B model takes the shared memory down by ~45 GB for a moment beside the site's engine
for i in $(seq 1 180); do
  avail=$(free -g | awk 'NR==2{print $7}')
  [ "$avail" -ge 45 ] && break
  [ $((i % 10)) -eq 1 ] && echo "$(date '+%F %T') waiting for memory: ${avail} GB free, need 45"
  sleep 60
done
avail=$(free -g | awk 'NR==2{print $7}')
if [ "$avail" -lt 45 ]; then echo "$(date '+%F %T') gave up: only ${avail} GB free after 3 hours"; exit 1; fi
echo "== $(date '+%F %T') v1, r64c with --guard (${avail} GB free)"
$PY evaluate.py --manifest data_test/test.jsonl --per-region 1000 --guard --out runs/eval_test_guard \
  --adapters "v1=runs/lora_v1/final,r64c=runs/lora_r64c/final" || exit 1
echo "== $(date '+%F %T') scoring"
$PY score_eval.py runs/eval_test_guard/v1 runs/eval_test_guard/r64c --base v1 \
  --out runs/eval_test_guard/report.json > runs/eval_test_guard/report.txt 2>&1
echo "== $(date '+%F %T') done (runs/eval_test_guard/report.txt)"

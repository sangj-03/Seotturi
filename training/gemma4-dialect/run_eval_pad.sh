#!/bin/bash
# Are the Jeju clips cut too tight? About 30% of the Jeju test clips start or end inside speech (1-5% in the
# other regions) and score worse. The same 1,000 Jeju utterances of data_test cut again with 0.3 s and 0.6 s
# of the recording kept on each side (prepare_data.py --only data_test/test.jsonl --pad), decoded by the
# served lora_r64c as the site does (--guard), against its run on the tight clips (runs/eval_test_guard/r64c).
#   setsid nohup ./run_eval_pad.sh >> runs/eval_pad/run.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
echo 1000 > /proc/self/oom_score_adj
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
REPO=$(cd ../.. && pwd -P)   # repository root
PY=${PY:-$REPO/.venv/bin/python}
for pad in 03 06; do
  until [ -s data_test_pad$pad/valid.jsonl ]; do sleep 30; done  # prepare_data.py writes it last
  # loading the 12B model takes the shared memory down by ~45 GB for a moment beside the site's engine
  for i in $(seq 1 360); do
    avail=$(free -g | awk 'NR==2{print $7}')
    [ "$avail" -ge 45 ] && break
    [ $((i % 10)) -eq 1 ] && echo "$(date '+%F %T') waiting for memory: ${avail} GB free, need 45"
    sleep 60
  done
  avail=$(free -g | awk 'NR==2{print $7}')
  if [ "$avail" -lt 45 ]; then echo "$(date '+%F %T') gave up: only ${avail} GB free after 6 hours"; exit 1; fi
  echo "== $(date '+%F %T') r64c on Jeju clips padded 0.${pad#0} s (${avail} GB free)"
  $PY evaluate.py --manifest data_test_pad$pad/valid.jsonl --per-region 1000 --guard --out runs/eval_pad \
    --adapters "pad$pad=runs/lora_r64c/final" || exit 1
done
echo "== $(date '+%F %T') scoring"
$PY score_eval.py runs/eval_test_guard/r64c runs/eval_pad/pad03 runs/eval_pad/pad06 --base r64c \
  --out runs/eval_pad/report.json > runs/eval_pad/report.txt 2>&1
echo "== $(date '+%F %T') done (runs/eval_pad/report.txt)"

#!/usr/bin/env bash
# Stage 5 (lora_r64, 2026-10-10): more data in the same rank-16 adapter stopped helping. lora_v2r (+100k
# utterances, lr 1e-4) and lora_v3 (+16k, lr 2e-5) both tie lora_v1 on the held-out data_test
# (runs/eval_test/report.txt), and training loss sits at validation loss (~0.28-0.30), so the adapter looks
# capacity-bound rather than overfit. This stage merges lora_v1 into the base weights and trains a fresh
# rank-64 LoRA on top: new utterances from data_more (not in data_v3), 12,000 per region, lr 2e-5 cosine,
# 3,000 steps of 16 (a first try at lr 1e-4 pushed validation loss from 0.281 to 0.350 by step 500:
# runs/lora_r64_lr1e-4, runs/stage5_r64_lr1e-4.log). Its validation loss (same 160 utterances) compares directly with lora_v1's 0.281.
# Then the final adapter on data_all/valid.jsonl and data_test, scored against v1.
#   setsid nohup ./keep_stage5_r64.sh > /dev/null 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")"
echo 1000 > /proc/self/oom_score_adj  # under memory pressure the kernel kills training, not the site's engine
PY=/data/seotturi/.venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_GPU_MEM_GB=${TRAIN_GPU_MEM_GB:-38}
step() { echo "$(date '+%F %T') == $*"; }

if [ ! -s data_r64/train.jsonl ]; then
    step "draw the stage-5 utterances from data_more (not in data_v3)"
    $PY - <<'PY'
import json, os, random
PER_REGION = 12000
used = {json.loads(line)["id"] for line in open("data_v3/train.jsonl", encoding="utf-8")}
src = "data_more/train.jsonl"
base = os.path.dirname(os.path.abspath(src))
rows = {}
for line in open(src, encoding="utf-8"):
    r = json.loads(line)
    if r["id"] not in used:
        rows.setdefault(r["region"], []).append(r)
random.seed(6)
out = [{**r, "audio": os.path.join(base, r["audio"])} for region in sorted(rows)
       for r in random.sample(rows[region], PER_REGION)]
random.shuffle(out)
os.makedirs("data_r64", exist_ok=True)
with open("data_r64/train.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(out), {k: sum(r["region"] == k for r in out) for k in sorted(rows)})
PY
fi

if [ ! -d runs/lora_r64/final ]; then
    step "train a rank-64 LoRA on top of the merged lora_v1 (lr 2e-5, 3,000 steps)"
    resume=()
    ls -d runs/lora_r64/checkpoint-* >/dev/null 2>&1 && resume=(--resume)
    $PY train.py --format basic --train data_r64/train.jsonl --valid data_all/valid.jsonl --out runs/lora_r64 \
        --merge-adapter runs/lora_v1/final --lora-r 64 --lora-alpha 128 --lr 2e-5 --max-steps 3000 \
        --save-steps 500 --eval-steps 250 "${resume[@]}"
fi

wait_for_memory() {
    # loading the 12B model takes the shared memory down by ~45 GB for a moment beside the site's engine
    for i in $(seq 1 180); do
        avail=$(free -g | awk 'NR==2{print $7}')
        [ "$avail" -ge 45 ] && return 0
        [ $((i % 10)) -eq 1 ] && step "waiting for memory: ${avail} GB free, need 45"
        sleep 60
    done
    return 1
}
if [ ! -s runs/eval_full/report_r64.txt ]; then
    wait_for_memory
    step "evaluate lora_r64 on data_all/valid.jsonl"
    $PY evaluate.py --manifest data_all/valid.jsonl --out runs/eval_full --merge-adapter runs/lora_v1/final \
        --adapters "r64=runs/lora_r64/final"
    $PY score_eval.py runs/eval_full/v1 runs/eval_full/r64 --base v1 \
        --out runs/eval_full/report_r64.json > runs/eval_full/report_r64.txt 2>&1
fi
if [ ! -s runs/eval_test/report_r64.txt ]; then
    wait_for_memory
    step "evaluate lora_r64 on data_test"
    $PY evaluate.py --manifest data_test/test.jsonl --per-region 1000 --out runs/eval_test \
        --merge-adapter runs/lora_v1/final --adapters "r64=runs/lora_r64/final"
    $PY score_eval.py runs/eval_test/v1 runs/eval_test/r64 --base v1 \
        --out runs/eval_test/report_r64.json > runs/eval_test/report_r64.txt 2>&1
fi
step "stage 5 (lora_r64) done"

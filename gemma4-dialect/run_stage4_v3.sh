#!/usr/bin/env bash
# Stage 4 (lora_v3, 2026-10-08): stage 3 (lora_v2r, region format, lr 1e-4) did not beat lora_v1 on the
# benchmark after 6,250 steps, so it was stopped. This is a short, targeted fine-tune of lora_v1 instead:
# the basic two-line format the app uses, lr 2e-5 (cosine), 4,000 steps of 16 utterances drawn from
# data_more (new to lora_v1) weighted to v1's weak regions: 제주 40 %, 강원 25 %, 충청 25 %, 경상 10 %.
# The web engine scores every new checkpoint against lora_v1 (SEOTTURI_WATCH lora_v3:basic).
#   setsid nohup ./keep_stage4_v3.sh > /dev/null 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")"
PY=/data/seotturi/.venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_GPU_MEM_GB=${TRAIN_GPU_MEM_GB:-38}
step() { echo "$(date '+%F %T') == $*"; }

if [ ! -s data_v3/train.jsonl ]; then
    step "draw the stage-4 utterances from data_more"
    $PY - <<'PY'
import json, os, random
SHARE = {"제주": 25600, "강원": 16000, "충청": 16000, "경상": 6400}
src = "data_more/train.jsonl"
base = os.path.dirname(os.path.abspath(src))
rows = {}
for line in open(src, encoding="utf-8"):
    r = json.loads(line)
    rows.setdefault(r["region"], []).append(r)
random.seed(4)
out = []
for region, n in SHARE.items():
    for r in random.sample(rows[region], n):
        out.append({**r, "audio": os.path.join(base, r["audio"])})
random.shuffle(out)
os.makedirs("data_v3", exist_ok=True)
with open("data_v3/train.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(out), {k: sum(r["region"] == k for r in out) for k in SHARE})
PY
fi

if [ ! -d runs/lora_v3/final ]; then
    step "fine-tune lora_v1 on data_v3 (basic format, lr 2e-5, 4,000 steps)"
    resume=()
    ls -d runs/lora_v3/checkpoint-* >/dev/null 2>&1 && resume=(--resume)
    $PY train.py --format basic --train data_v3/train.jsonl --valid data_all/valid.jsonl --out runs/lora_v3 \
        --init-adapter runs/lora_v1/final --lr 2e-5 --max-steps 4000 --save-steps 250 --eval-steps 250 "${resume[@]}"
fi
step "stage 4 (lora_v3) done"

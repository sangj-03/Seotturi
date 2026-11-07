#!/usr/bin/env bash
# Stage 6 (lora_s6, 2026-10-11): capacity was the one change that helped. The rank-64 LoRA on top of the merged
# lora_v1 beat v1 on both held-out sets, and decoded as the site does (--guard) the served combination
# lora_r64c beats v1 on data_test: dialect 0.147 -> 0.145 (p_better 0.98), standard 0.153 -> 0.151 (0.96),
# Jeju 0.308 -> 0.302 (runs/eval_test_guard/report.txt). Jeju is still the weak region (~30% CER), and about
# 30% of its clips start or end inside speech: the AI Hub label times cut words off (1-5% in the other regions).
# So this stage adds another rank-64 LoRA on top of the merged lora_r64c (v1 + r64), trained on new utterances,
# 16,000 per region, lr 2e-5 cosine, 4,000 steps of 16, cut with the padding run_eval_pad.sh found to help
# (none if it did not). Then lora_r64c + lora_s6 are folded into one adapter for the site (lora_s6c, rank 144)
# and scored against lora_r64c on data_test with --guard (and on the padded data_test when there is padding).
#   setsid nohup ./keep_stage6.sh > /dev/null 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")"
echo 1000 > /proc/self/oom_score_adj  # under memory pressure the kernel kills training, not the site's engine
PY=/data/seotturi/.venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_GPU_MEM_GB=${TRAIN_GPU_MEM_GB:-38}
step() { echo "$(date '+%F %T') == $*"; }
wait_for_memory() {
    # loading the 12B model takes the shared memory down by ~45 GB for a moment beside the site's engine
    # (and, since 2026-10-11, beside other GPU jobs on this machine)
    for i in $(seq 1 720); do
        avail=$(free -g | awk 'NR==2{print $7}')
        [ "$avail" -ge 45 ] && return 0
        [ $((i % 30)) -eq 1 ] && step "waiting for memory: ${avail} GB free, need 45"
        sleep 60
    done
    return 1
}

if [ ! -s runs/stage6_pad.txt ]; then
    while [ ! -s runs/eval_pad/report.json ]; do sleep 300; done
    # the padding, if any, that made the served model's Jeju transcripts better beyond doubt
    $PY - > runs/stage6_pad.txt <<'PY'
import json
jeju = json.load(open("runs/eval_pad/report.json", encoding="utf-8"))["regions"]["제주"]
ok = [(jeju["dialect_cer"][k]["value"], k) for k in ("pad03", "pad06")
      if jeju["dialect_cer"][k]["p_better"] >= 0.95 and jeju["standard_cer"][k]["p_better"] >= 0.5]
print({"pad03": "0.3", "pad06": "0.6"}[min(ok)[1]] if ok else "0")
PY
fi
PAD=$(cat runs/stage6_pad.txt)
step "padding ${PAD} s"

if [ ! -s data_s6/train.jsonl ]; then
    if [ "$PAD" = "0" ]; then
        step "draw the stage-6 utterances from data_more (not in data_v3 or data_r64)"
        $PY - <<'PY'
import json, os, random
PER_REGION = 16000
used = {json.loads(line)["id"] for p in ("data_v3/train.jsonl", "data_r64/train.jsonl")
        for line in open(p, encoding="utf-8")}
src = "data_more/train.jsonl"
base = os.path.dirname(os.path.abspath(src))
rows = {}
for line in open(src, encoding="utf-8"):
    r = json.loads(line)
    if r["id"] not in used:
        rows.setdefault(r["region"], []).append(r)
random.seed(7)
out = [{**r, "audio": os.path.join(base, r["audio"])} for region in sorted(rows)
       for r in random.sample(rows[region], PER_REGION)]
random.shuffle(out)
os.makedirs("data_s6", exist_ok=True)
with open("data_s6/train.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(out), {k: sum(r["region"] == k for r in out) for k in sorted(rows)})
PY
    else
        step "cut 16,000 new utterances per region with ${PAD} s padding (not in data_all or data_r64)"
        rm -rf data_s6
        $PY prepare_data.py --datasets 118,119,121,122 --train-per-region 16000 --valid-per-region 0 \
            --exclude data_all/train.jsonl,data_r64/train.jsonl --pad "$PAD" --seed 7 --out data_s6.tmp
        $PY - <<'PY'
import json, random
rows = [json.loads(line) for line in open("data_s6.tmp/train.jsonl", encoding="utf-8")]
random.seed(7)
random.shuffle(rows)  # prepare_data.py sorts by id, so one region would follow another
with open("data_s6.tmp/train.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
PY
        mv data_s6.tmp data_s6
    fi
fi
if [ "$PAD" != "0" ] && [ ! -s data_test_pad/valid.jsonl ]; then
    step "cut data_test again with ${PAD} s padding"
    rm -rf data_test_pad
    $PY prepare_data.py --datasets 118,119,121,122 --train-per-region 0 --valid-per-region 0 \
        --only data_test/test.jsonl --pad "$PAD" --out data_test_pad.tmp
    mv data_test_pad.tmp data_test_pad
fi

if [ ! -d runs/lora_s6/final ]; then
    wait_for_memory
    step "train a rank-64 LoRA on top of the merged lora_r64c (lr 2e-5, 4,000 steps)"
    resume=()
    ls -d runs/lora_s6/checkpoint-* >/dev/null 2>&1 && resume=(--resume)
    $PY train.py --format basic --train data_s6/train.jsonl --valid data_all/valid.jsonl --out runs/lora_s6 \
        --merge-adapter runs/lora_r64c/final --lora-r 64 --lora-alpha 128 --lr 2e-5 --max-steps 4000 \
        --save-steps 500 --eval-steps 250 "${resume[@]}"
fi
[ -d runs/lora_s6c/final ] || $PY combine_adapters.py runs/lora_r64c/final runs/lora_s6/final runs/lora_s6c/final

if [ ! -s runs/eval_test_guard/report_s6c.txt ]; then
    wait_for_memory
    step "evaluate lora_s6c on data_test as the site decodes"
    $PY evaluate.py --manifest data_test/test.jsonl --per-region 1000 --guard --out runs/eval_test_guard \
        --adapters "s6c=runs/lora_s6c/final"
    $PY score_eval.py runs/eval_test_guard/v1 runs/eval_test_guard/r64c runs/eval_test_guard/s6c --base r64c \
        --out runs/eval_test_guard/report_s6c.json > runs/eval_test_guard/report_s6c.txt 2>&1
fi
if [ "$PAD" != "0" ] && [ ! -s runs/eval_test_pad/report.txt ]; then
    wait_for_memory
    step "evaluate lora_r64c and lora_s6c on the padded data_test"
    $PY evaluate.py --manifest data_test_pad/valid.jsonl --per-region 1000 --guard --out runs/eval_test_pad \
        --adapters "r64c=runs/lora_r64c/final,s6c=runs/lora_s6c/final"
    $PY score_eval.py runs/eval_test_pad/r64c runs/eval_test_pad/s6c --base r64c \
        --out runs/eval_test_pad/report.json > runs/eval_test_pad/report.txt 2>&1
fi
step "stage 6 (lora_s6) done"

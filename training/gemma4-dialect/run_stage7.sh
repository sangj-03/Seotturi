#!/usr/bin/env bash
# Stage 7 (lora_s7, 2026-10-12): the user's goal is a single-digit CER. The error analysis of the served
# lora_s6c (error_analysis.py, README "Error analysis") shows that perfect sentence edges, dialect words and
# labels would still leave the CER near the error rate of ordinary words in the middle of a sentence (0.10-0.11):
# the recognition itself has to improve, above all in Jeju and Gangwon. What helped so far was capacity and new
# data (stages 5 and 6), so this stage trains a twice as large LoRA (rank 128) on top of the merged lora_s6c, on
# about five times the new data of stage 6: every data_more utterance the served lineage (v1 -> r64 -> s6) has
# not seen, all of Jeju and Gangwon and part of Chungcheong and Gyeongsang (lora_v3's draw is fine: that branch
# is not in the lineage), without clips whose label cannot fit their audio (fewer than 2 or more than 8
# syllables a second: 5 % of data_test, mean CER 0.33, nine times the share of misaligned labels).
# lr 2e-5 cosine, one pass (~19,000 steps of 16, about three days at ~13 s a step), a checkpoint every 500 steps.
# At steps 5,000, 10,000 and 15,000 training pauses for a check on 1,000 clips of data_test (250 per region,
# decoded as the site does) against lora_s6c on the same clips: runs/eval_s7/ck<step>/report.txt.
# Then lora_s6c + lora_s7 are folded into one adapter for the site (lora_s7c, rank 272) and scored against
# lora_s6c on all of data_test with --guard, and error_analysis.py runs on it.
#
# Start it from cron, not from a terminal: systemd-oomd kills a terminal's whole scope (and everything started
# from it) under memory pressure, which took the site's engine down on 2026-10-12. keep_stage7.sh restarts it.
set -euo pipefail
cd "$(dirname "$0")"
echo 1000 > /proc/self/oom_score_adj  # under memory pressure the kernel kills training, not the site's engine
REPO=$(cd ../.. && pwd -P)   # repository root
PY=${PY:-$REPO/.venv/bin/python}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# rank 128 keeps about 4 GB more optimizer state than rank 64 (peak 33.6 GB in stage 6)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_GPU_MEM_GB=${TRAIN_GPU_MEM_GB:-42}
OUT=runs/lora_s7
INTERIM=(5000 10000 15000)
step() { echo "$(date '+%F %T') == $*"; }
wait_for_memory() {
    # loading the 12B model takes the shared memory down by ~45 GB for a moment beside the site's engine
    for i in $(seq 1 720); do
        avail=$(free -g | awk 'NR==2{print $7}')
        [ "$avail" -ge 45 ] && return 0
        [ $((i % 30)) -eq 1 ] && step "waiting for memory: ${avail} GB free, need 45"
        sleep 60
    done
    return 1
}
newest() { ls -d $OUT/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1 || true; }  # ls fails with none
complete() {  # checkpoint $1 written in full at least a minute ago (trainer_state.json comes last)
    local f=$OUT/checkpoint-$1/trainer_state.json
    [ -s "$f" ] && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -ge 60 ]
}
interim() {  # checkpoint $1 against lora_s6c on the first 250 clips per region of data_test
    # (called as a condition, so set -e is off in here: every step returns on failure itself)
    local s=$1 d=runs/eval_s7/ck$1
    wait_for_memory || return 1
    step "interim check: checkpoint-$s on 1,000 clips of data_test, decoded as the site does"
    $PY evaluate.py --manifest data_test/test.jsonl --per-region 250 --guard --merge-adapter runs/lora_s6c/final \
        --adapter $OUT/checkpoint-$s --out $d || return 1
    $PY score_eval.py runs/eval_test_guard/s6c $d --base s6c --out $d/report.json > $d/report.txt 2>&1 || return 1
    step "interim check $s done: $d/report.txt"
}

echo lora_s7 > runs/training_now  # the site's benchmark page shows this run's progress (seotturi-web app.py)

if [ ! -s data_s7/train.jsonl ]; then
    step "draw the stage-7 utterances from data_more (new to v1 -> r64 -> s6, 2-8 syllables a second)"
    $PY - <<'PY'
import json, os, random, re
TAKE = {"제주": None, "강원": None, "충청": 72000, "경상": 56000}  # None: every one left
HANGUL = re.compile(r"[가-힣]")
seen = set()
for p in ("data_118/train.jsonl", "data_all/train.jsonl", "data_r64/train.jsonl", "data_s6/train.jsonl"):
    seen.update(json.loads(line)["id"] for line in open(p, encoding="utf-8"))
src = "data_more/train.jsonl"
base = os.path.dirname(os.path.abspath(src))
rows = {}
for line in open(src, encoding="utf-8"):
    r = json.loads(line)
    rate = len(HANGUL.findall(r["dialect_form"])) / max(r["duration"], 0.1)
    if r["id"] not in seen and 2.0 <= rate <= 8.0:
        rows.setdefault(r["region"], []).append(r)
random.seed(8)
out = []
for region, n in TAKE.items():
    pick = rows[region] if n is None else random.sample(rows[region], n)
    out += [{**r, "audio": os.path.join(base, r["audio"])} for r in pick]
random.shuffle(out)
os.makedirs("data_s7", exist_ok=True)
with open("data_s7/train.jsonl.tmp", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
os.replace("data_s7/train.jsonl.tmp", "data_s7/train.jsonl")
print(len(out), {k: sum(r["region"] == k for r in out) for k in TAKE})
PY
fi

while [ ! -d $OUT/final ]; do
    have=$(newest)
    todo=""
    for s in "${INTERIM[@]}"; do
        [ -e runs/eval_s7/ck$s/report.json ] || [ -e runs/eval_s7/ck$s/failed ] && continue
        [ "$s" -lt "${have:-0}" ] && [ ! -d $OUT/checkpoint-$s ] && continue  # rotated out (only 3 are kept)
        todo=$s
        break
    done
    if [ -n "$todo" ] && [ -d $OUT/checkpoint-$todo ] && complete "$todo"; then
        if ! interim "$todo"; then  # a failed check must not stop the training
            step "interim check $todo failed, going on"
            mkdir -p runs/eval_s7/ck$todo && touch runs/eval_s7/ck$todo/failed
        fi
        continue
    fi
    wait_for_memory
    resume=()
    [ -n "$have" ] && resume=(--resume)
    step "train a rank-128 LoRA on top of the merged lora_s6c (lr 2e-5, one pass), from checkpoint ${have:-none}"
    $PY train.py --format basic --train data_s7/train.jsonl --valid data_all/valid.jsonl --out $OUT \
        --merge-adapter runs/lora_s6c/final --lora-r 128 --lora-alpha 256 --lr 2e-5 --epochs 1 \
        --save-steps 500 --eval-steps 1000 "${resume[@]}" &
    pid=$!
    if [ -n "$todo" ]; then  # pause at the next interim checkpoint, check it, then go on from there
        while kill -0 $pid 2>/dev/null && ! { [ -d $OUT/checkpoint-$todo ] && complete "$todo"; }; do sleep 60; done
        if kill -0 $pid 2>/dev/null; then
            step "checkpoint-$todo written: pausing the training for the interim check"
            kill $pid
            wait $pid || true
            continue
        fi
    fi
    rc=0
    wait $pid || rc=$?
    [ $rc -eq 0 ] || exit $rc
done

[ -d runs/lora_s7c/final ] || $PY combine_adapters.py runs/lora_s6c/final $OUT/final runs/lora_s7c/final

if [ ! -s runs/eval_test_guard/report_s7c.txt ]; then
    wait_for_memory
    step "evaluate lora_s7c on data_test as the site decodes"
    $PY evaluate.py --manifest data_test/test.jsonl --per-region 1000 --guard --out runs/eval_test_guard \
        --adapters "s7c=runs/lora_s7c/final"
    $PY score_eval.py runs/eval_test_guard/v1 runs/eval_test_guard/r64c runs/eval_test_guard/s6c \
        runs/eval_test_guard/s7c --base s6c \
        --out runs/eval_test_guard/report_s7c.json > runs/eval_test_guard/report_s7c.txt 2>&1
    $PY error_analysis.py runs/eval_test_guard/s7c > /dev/null
fi
rm -f runs/training_now
step "stage 7 (lora_s7) done"

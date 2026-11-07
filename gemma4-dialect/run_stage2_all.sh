#!/usr/bin/env bash
# Stage 2: once the download and stage 1 are both finished, continue the Gangwon
# adapter on Gyeongsang (119), Jeju (121) and Chungcheong (122), with 10% Gangwon
# replay so the first region is not forgotten, then score base vs. adapter on all
# four regions. Jeolla (120) stays out: its labels have no timestamps.
set -euo pipefail
cd "$(dirname "$0")"
PY=/data/seotturi/.venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
step() { echo "$(date '+%F %T') == $*"; }
running() { pgrep -f "^bash \./$1" >/dev/null; }

downloads_done() {
    running download_dialect.sh && return 1
    for d in 118 119 120 121 122; do
        local want have
        want=$(wc -l < "/data/aihub/$d/filekeys.txt" 2>/dev/null || echo 0)
        have=$(ls "/data/aihub/$d/.done" 2>/dev/null | wc -l)
        [ "$want" -gt 0 ] && [ "$have" -eq "$want" ] || return 1
    done
}

step "waiting for the download and stage 1"
until downloads_done && [ -s runs/lora_gangwon_eval/summary.json ] && ! running run_stage1_gangwon.sh; do
    sleep 60
done

if [ ! -s data_rest/train.jsonl ]; then
    step "prepare Gyeongsang, Jeju and Chungcheong data"
    $PY prepare_data.py --datasets 119,121,122 --train-per-region 25000 --valid-per-region 300 --out data_rest
fi

if [ ! -s data_all/train.jsonl ]; then
    step "merge manifests (7,500 Gangwon replay utterances)"
    $PY - <<'EOF'
import json, os, random
def rows(path):
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        r["audio"] = os.path.join(base, r["audio"])  # absolute, so one manifest can mix folders
        out.append(r)
    return out
os.makedirs("data_all", exist_ok=True)
rng = random.Random(0)
train = rows("data_rest/train.jsonl") + rng.sample(rows("data_118/train.jsonl"), 7500)
rng.shuffle(train)
valid = rows("data_118/valid.jsonl") + rows("data_rest/valid.jsonl")
for name, rs in (("train", train), ("valid", valid)):
    with open(f"data_all/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rs))
EOF
fi

if [ ! -s runs/base_eval/summary.json ]; then
    step "evaluate base model on all four regions"
    $PY evaluate.py --manifest data_all/valid.jsonl --out runs/base_eval
fi

if [ ! -d runs/lora_v1/final ]; then
    step "continue training on all regions"
    resume=()
    ls -d runs/lora_v1/checkpoint-* >/dev/null 2>&1 && resume=(--resume)
    $PY train.py --train data_all/train.jsonl --valid data_all/valid.jsonl --out runs/lora_v1 --epochs 1 \
        --init-adapter runs/lora_gangwon/final "${resume[@]}"
fi

if [ ! -s runs/lora_v1_eval/summary.json ]; then
    step "evaluate final adapter on all four regions"
    $PY evaluate.py --manifest data_all/valid.jsonl --adapter runs/lora_v1/final --out runs/lora_v1_eval
fi

step "compare"
$PY - <<'EOF'
import json
b = json.load(open("runs/base_eval/summary.json"))
t = json.load(open("runs/lora_v1_eval/summary.json"))
rows = [("overall", b["overall"], t["overall"])] + [(k, b["regions"][k], t["regions"][k]) for k in b["regions"]]
lines = ["| | dialect CER base | dialect CER tuned | standard CER base | standard CER tuned |",
         "| --- | ---: | ---: | ---: | ---: |"]
for name, x, y in rows:
    lines.append(f"| {name} | {x['cer_dialect']:.3f} | {y['cer_dialect']:.3f} | {x['cer_standard']:.3f} | {y['cer_standard']:.3f} |")
open("runs/lora_v1_comparison.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
step "stage 2 done"

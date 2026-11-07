#!/usr/bin/env bash
# Stage 3: continue the stage-2 adapter (runs/lora_v1) on 500,000 more utterances,
# 125,000 per region (Gangwon, Gyeongsang, Jeju, Chungcheong), none of which were
# used in stages 1-2. Scored on the same 1,199 validation utterances as before.
# Each step is skipped when its output exists; training resumes from checkpoints.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd -P)   # repository root
PY=${PY:-$REPO/.venv/bin/python}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
step() { echo "$(date '+%F %T') == $*"; }

if [ ! -s data_more/train.jsonl ]; then
    step "prepare 500,000 new utterances"
    $PY prepare_data.py --datasets 118,119,121,122 --train-per-region 125000 --valid-per-region 0 \
        --out data_more --seed 1 --exclude data_118/train.jsonl,data_rest/train.jsonl
fi

if [ ! -d runs/lora_v2/final ]; then
    step "continue training from lora_v1"
    resume=()
    ls -d runs/lora_v2/checkpoint-* >/dev/null 2>&1 && resume=(--resume)
    $PY train.py --train data_more/train.jsonl --valid data_all/valid.jsonl --out runs/lora_v2 --epochs 1 \
        --init-adapter runs/lora_v1/final --save-steps 1000 --eval-steps 2000 "${resume[@]}"
fi

if [ ! -s runs/lora_v2_eval/summary.json ]; then
    step "evaluate lora_v2 on all four regions"
    $PY evaluate.py --manifest data_all/valid.jsonl --adapter runs/lora_v2/final --out runs/lora_v2_eval
fi

step "compare"
$PY - <<'EOF'
import json
runs = [("base", "runs/base_eval"), ("v1 (100k)", "runs/lora_v1_eval"), ("v2 (+500k)", "runs/lora_v2_eval")]
s = {name: json.load(open(f"{path}/summary.json")) for name, path in runs}
names = [n for n, _ in runs]
lines = ["| | " + " | ".join(f"dialect {n}" for n in names) + " | " + " | ".join(f"standard {n}" for n in names) + " |",
         "| --- |" + " ---: |" * (2 * len(names))]
for key in ["overall"] + list(s["base"]["regions"]):
    get = (lambda d: d["overall"]) if key == "overall" else (lambda d, k=key: d["regions"][k])
    lines.append(f"| {key} | " + " | ".join(f"{get(s[n])['cer_dialect']:.3f}" for n in names) + " | "
                 + " | ".join(f"{get(s[n])['cer_standard']:.3f}" for n in names) + " |")
open("runs/lora_v2_comparison.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
step "stage 3 done"

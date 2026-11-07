#!/usr/bin/env bash
# Full pipeline after the AI Hub download finishes. Each step is skipped when its
# output already exists, and training resumes from its last checkpoint, so the
# script can simply be rerun after an interruption.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd -P)   # repository root
PY=${PY:-$REPO/.venv/bin/python}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# 120 (Jeolla) is left out: its labels are plain text without timestamps or
# standard-Korean forms, so its utterances cannot be cut out of the recordings.
DATASETS=${DATASETS:-118,119,121,122}
PER_REGION=${PER_REGION:-25000}
RUN=${RUN:-lora_v1}

step() { echo "$(date '+%F %T') == $*"; }

if [ ! -s data/train.jsonl ]; then
    step "prepare data ($DATASETS, $PER_REGION per region)"
    $PY prepare_data.py --datasets "$DATASETS" --train-per-region "$PER_REGION" --valid-per-region 300
fi

if [ ! -s runs/base_eval/summary.json ]; then
    step "evaluate base model"
    $PY evaluate.py --out runs/base_eval
fi

if [ ! -d "runs/$RUN/final" ]; then
    step "train $RUN"
    resume=()
    ls -d "runs/$RUN"/checkpoint-* >/dev/null 2>&1 && resume=(--resume)
    $PY train.py --out "runs/$RUN" --epochs 1 "${resume[@]}"
fi

if [ ! -s "runs/${RUN}_eval/summary.json" ]; then
    step "evaluate $RUN"
    $PY evaluate.py --adapter "runs/$RUN/final" --out "runs/${RUN}_eval"
fi

step "compare"
$PY - "$RUN" <<'EOF'
import json, sys
run = sys.argv[1]
b = json.load(open("runs/base_eval/summary.json"))
t = json.load(open(f"runs/{run}_eval/summary.json"))
rows = [("overall", b["overall"], t["overall"])] + [(k, b["regions"][k], t["regions"][k]) for k in b["regions"]]
lines = ["| | dialect CER base | dialect CER tuned | standard CER base | standard CER tuned |", "| --- | ---: | ---: | ---: | ---: |"]
for name, x, y in rows:
    lines.append(f"| {name} | {x['cer_dialect']:.3f} | {y['cer_dialect']:.3f} | {x['cer_standard']:.3f} | {y['cer_standard']:.3f} |")
open(f"runs/{run}_comparison.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
step "done"

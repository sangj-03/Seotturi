#!/usr/bin/env bash
# Score the base model, lora_v1 and (once it exists) lora_v2r on the intonation test set
# (prepare_intonation.py) and write runs/intonation_comparison.md.
# Uses the GPU for about an hour per model. Each step is skipped when its output exists.
set -euo pipefail
cd "$(dirname "$0")"
PY=/data/seotturi/.venv/bin/python
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
step() { echo "$(date '+%F %T') == $*"; }

if [ ! -s data_intonation/test.jsonl ]; then
    step "build the intonation test set"
    $PY prepare_intonation.py --out data_intonation
fi

runs=()
# name:adapter:format
for spec in base::basic v1:runs/lora_v1/final:basic v2r:runs/lora_v2r/final:region; do
    IFS=: read -r name adapter fmt <<< "$spec"
    if [ -n "$adapter" ] && [ ! -d "$adapter" ]; then
        continue
    fi
    out=runs/intonation_$name
    if [ ! -s "$out/summary.json" ]; then
        step "evaluate $name on the intonation set"
        $PY evaluate.py --format "$fmt" --manifest data_intonation/test.jsonl --per-region 1000000 --out "$out" \
            ${adapter:+--adapter "$adapter"}
    fi
    runs+=("$name=$out")
done

step "score"
$PY score_intonation.py data_intonation/test.jsonl "${runs[@]}" | tee runs/intonation_comparison.md
step "intonation eval done"

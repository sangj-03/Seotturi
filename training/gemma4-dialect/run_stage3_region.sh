#!/usr/bin/env bash
# Stage 3 (region format): continue runs/lora_v1/final on data_more plus 44,000 utterances of
# Seoul-Gyeonggi and Jeolla speakers, with a `지역:` line (the speaker's accent zone, from
# birthplace and residence) in front of the answer, so the model has to listen to accent and
# intonation, not only the words. Replaces run_stage3_more.sh, stopped on 2026-10-06 at step 4,949.
# Each step is skipped when its output exists; training resumes from checkpoints.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd -P)   # repository root
PY=${PY:-$REPO/.venv/bin/python}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# Batches of varying audio length fragment PyTorch's GPU cache: without these the run reserved 44 GB for
# ~31 GB in use and pushed the app's engine out of memory (2026-10-07 19:27). Shared memory, so keep it capped.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_GPU_MEM_GB=${TRAIN_GPU_MEM_GB:-38}
step() { echo "$(date '+%F %T') == $*"; }

if [ ! -s data_more/train_origin.jsonl ] || [ ! -s data_zone_extra/train.jsonl ]; then
    step "label speaker zones and add Seoul-Gyeonggi / Jeolla speakers"
    $PY prepare_origin.py data_more/train.jsonl data_all/valid.jsonl --extra-out data_zone_extra
fi
if ! grep -q '"subset": "origin"' data_intonation/test.jsonl 2>/dev/null; then
    step "build the intonation test set"
    $PY prepare_intonation.py --out data_intonation
fi

if [ ! -d runs/lora_v2r/final ]; then
    step "continue training from lora_v1 with the region format"
    resume=()
    ls -d runs/lora_v2r/checkpoint-* >/dev/null 2>&1 && resume=(--resume)
    $PY train.py --format region --train data_more/train_origin.jsonl,data_zone_extra/train.jsonl \
        --valid data_all/valid_origin.jsonl --out runs/lora_v2r --epochs 1 \
        --init-adapter runs/lora_v1/final --save-steps 250 --eval-steps 2000 "${resume[@]}"
fi

if [ ! -s runs/lora_v2r_eval/summary.json ]; then
    step "evaluate lora_v2r on all four regions"
    $PY evaluate.py --format region --manifest data_all/valid_origin.jsonl --adapter runs/lora_v2r/final \
        --out runs/lora_v2r_eval
fi

step "compare"
$PY - <<'EOF'
import json
runs = [("base", "runs/base_eval"), ("v1 (100k)", "runs/lora_v1_eval"), ("v2r (+500k, 지역)", "runs/lora_v2r_eval")]
s = {name: json.load(open(f"{path}/summary.json")) for name, path in runs}
names = [n for n, _ in runs]
lines = ["| | " + " | ".join(f"dialect {n}" for n in names) + " | " + " | ".join(f"standard {n}" for n in names) + " |",
         "| --- |" + " ---: |" * (2 * len(names))]
for key in ["overall"] + list(s["base"]["regions"]):
    get = (lambda d: d["overall"]) if key == "overall" else (lambda d, k=key: d["regions"][k])
    lines.append(f"| {key} | " + " | ".join(f"{get(s[n])['cer_dialect']:.3f}" for n in names) + " | "
                 + " | ".join(f"{get(s[n])['cer_standard']:.3f}" for n in names) + " |")
z = s[names[-1]]
lines += ["", f"Zone accuracy of {names[-1]}: {z['zone']['accuracy']:.1%} of {z['zone']['n']} utterances, "
          f"{z['zone_plain_speech']['accuracy']:.1%} of {z['zone_plain_speech']['n']} without dialect words."]
open("runs/lora_v2r_comparison.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF

step "intonation eval"
./run_intonation_eval.sh
step "stage 3 (region) done"

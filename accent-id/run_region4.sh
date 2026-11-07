#!/bin/bash
# 4-region run (region4.py): waits for `features.py r4_dev_aug r4_aug`, then assembles the r4 splits from
# existing features, trains the head and scores it under the site's recording conditions.
#   setsid nohup ./run_region4.sh > runs/region4.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/data/seotturi/.venv/bin/python

until [ -f features/r4_aug.npy ]; do
  if ! pgrep -f '^/data/seotturi/.venv/bin/python features.py r4' > /dev/null; then
    echo "== features.py stopped before features/r4_aug.npy was written"; exit 1
  fi
  sleep 60
done
echo "== $(date '+%F %T') features done"
$PY assemble_big.py --split data_r4/train --name r4_train --sources train_big,r4_aug
$PY assemble_big.py --split data_r4/dev --name r4_dev --sources dev,r4_dev_aug
for s in test_plain test_same test_dialect bench; do
  $PY assemble_big.py --split "data_r4/$s" --name "r4_$s" --sources "$s"
done
echo "== $(date '+%F %T') assembled"
$PY train_head.py --zones region4 --train r4_train --dev r4_dev \
  --tests r4_test_plain,r4_test_same,r4_test_dialect,r4_bench --out runs/region4
echo "== $(date '+%F %T') trained"
$PY conditions.py runs/region4
echo "== $(date '+%F %T') done"

#!/bin/bash
# Standard-vs-regional detector (region4.py --std2): features of the Seoul copies, assembly, the head with and
# without copies, and the standard-vs-regional comparison under the site's conditions with the served model.
#   setsid nohup ./run_std2.sh > runs/std2.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/data/seotturi/.venv/bin/python
TESTS=std2_test_plain,std2_test_same,std2_test_dialect,std2_bench

$PY features.py std_dev_aug std_aug
echo "== $(date '+%F %T') features"
$PY assemble_big.py --split data_std2/train --name std2_train --sources train_big,r4_aug,std_aug
$PY assemble_big.py --split data_std2/dev --name std2_dev --sources dev,r4_dev_aug,std_dev_aug
for s in test_plain test_same test_dialect bench; do
  $PY assemble_big.py --split "data_std2/$s" --name "std2_$s" --sources "$s"
done
mkdir -p data_std2/orig
$PY -c "import json; rows = [json.loads(l) for l in open('data_std2/train/train.jsonl', encoding='utf-8')]; \
open('data_std2/orig/train.jsonl', 'w', encoding='utf-8').writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows if 'aug' not in r)"
$PY assemble_big.py --split data_std2/orig --name std2_orig --sources train_big
echo "== $(date '+%F %T') assembled"
$PY train_head.py --zones std2 --train std2_train --dev std2_dev --tests $TESTS --out runs/std2
$PY train_head.py --zones std2 --train std2_orig --dev std2_dev --tests $TESTS --out runs/std2_noaug
echo "== $(date '+%F %T') trained"
echo "== conditions std2"; $PY conditions.py runs/std2
echo "== conditions std2_noaug"; $PY conditions.py runs/std2_noaug
echo "== conditions served ensemble_big (standard bonus 1.25)"; $PY conditions.py runs/ensemble_big --binary --standard-bias 1.25
echo "== $(date '+%F %T') done"

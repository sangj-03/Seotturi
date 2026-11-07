#!/bin/bash
# encode (own process), then draw; the watchdog stops either one if memory gets tight.
cd /data/seotturi/img-gen
export PYTHONPATH=/data/seotturi/img-gen/pylib HF_HUB_OFFLINE=1
PY=/data/seotturi/.venv/bin/python
( sleep 3; ./watchdog.sh >> watchdog.log 2>&1 ) &
$PY -u gen.py encode > encode.log 2>&1 || { echo "encode failed"; tail -5 encode.log; exit 1; }
echo "$(date +%T) encoded: $(grep -c '^encoded' encode.log) prompts"
sleep 5
( sleep 3; ./watchdog.sh >> watchdog.log 2>&1 ) &
$PY -u gen.py draw --seeds 2 "$@" > draw.log 2>&1
echo "$(date +%T) draw exit $?"; tail -3 draw.log

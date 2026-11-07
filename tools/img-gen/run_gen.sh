#!/bin/bash
# encode (own process), then draw; the watchdog stops either one if memory gets tight.
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/pylib" HF_HUB_OFFLINE=1
REPO=$(cd ../.. && pwd -P)   # repository root
PY=${PY:-$REPO/.venv/bin/python}
( sleep 3; ./watchdog.sh >> watchdog.log 2>&1 ) &
$PY -u gen.py encode > encode.log 2>&1 || { echo "encode failed"; tail -5 encode.log; exit 1; }
echo "$(date +%T) encoded: $(grep -c '^encoded' encode.log) prompts"
sleep 5
( sleep 3; ./watchdog.sh >> watchdog.log 2>&1 ) &
$PY -u gen.py draw --seeds 2 "$@" > draw.log 2>&1
echo "$(date +%T) draw exit $?"; tail -3 draw.log

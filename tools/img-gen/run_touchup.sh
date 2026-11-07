#!/bin/bash
# Second pass on the chosen drawings (touchup.json): encode in its own process, then draw; watchdog as before.
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/pylib" HF_HUB_OFFLINE=1
REPO=$(cd ../.. && pwd -P)   # repository root
PY=${PY:-$REPO/.venv/bin/python}
( sleep 3; ./watchdog.sh >> watchdog.log 2>&1 ) &
$PY -u gen.py encode --jobs touchup.json > encode2.log 2>&1 || { echo "encode failed"; tail -5 encode2.log; exit 1; }
echo "$(date +%T) encoded: $(grep -c '^encoded' encode2.log) prompts"
sleep 5
( sleep 3; ./watchdog.sh >> watchdog.log 2>&1 ) &
$PY -u gen.py draw --jobs touchup.json --seeds 2 > draw2.log 2>&1
echo "$(date +%T) draw exit $?"; grep -c 'seed' draw2.log

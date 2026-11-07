#!/usr/bin/env bash
# Start/stop the 서투리 web app: the Gemma 4 engine (GPU, 127.0.0.1:8788), the speech service
# (Supertonic 3 on the CPU, 127.0.0.1:8789), the accent classifier (GPU, 127.0.0.1:8790) and the web
# server (:8787).
#   ./run.sh start            start all four (the engine needs about 5 minutes to load the model)
#   ./run.sh stop             stop all four
#   ./run.sh restart-web      restart only the web server (the model stays loaded)
#   ./run.sh restart-tts      restart only the speech service
#   ./run.sh restart-accent   restart only the accent classifier (after changing accent-id/runs/served)
#   ./run.sh restart-engine   reload the model (about 5 minutes without AI)
#   ./run.sh status           served adapter, its score, next update check
# Each server runs under scripts/supervise.sh, which restarts it if it crashes.
# Environment: SEOTTURI_HOST (default 127.0.0.1; 0.0.0.0 to open it to the LAN), SEOTTURI_PORT (8787),
# SEOTTURI_SSL_CERT/KEY, SEOTTURI_UPDATE_HOURS (2), SEOTTURI_WATCH (lora_r64c:basic,lora_s6c:basic),
# SEOTTURI_FALLBACK (lora_v1:basic), SEOTTURI_PROMPT_LOOKUP (auto|0|1), SEOTTURI_TTS_THREADS (4).
set -euo pipefail
cd "$(dirname "$0")"
PY=/data/seotturi/.venv/bin/python       # training venv: torch, transformers, fastapi
TTS_PY=$PWD/.venv-tts/bin/python         # separate venv: supertonic, onnxruntime
mkdir -p logs state data
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

pid_of() { pgrep -f "^$2 $PWD/server/$1" || true; }   # script interpreter
sup_of() { pgrep -f "scripts/supervise.sh $1 " || true; }

start() {  # name script log interpreter
    if [ -n "$(pid_of "$2" "$4")" ] || [ -n "$(sup_of "$1")" ]; then echo "$1 already running"; return; fi
    rm -f "state/stop-$1"
    setsid nohup scripts/supervise.sh "$1" "$4" "$PWD/server/$2" >> "logs/$3" 2>&1 < /dev/null &
    echo "$1 starting (log: logs/$3)"
}
stop() {  # name script interpreter
    touch "state/stop-$1"
    local p
    p=$(pid_of "$2" "$3")
    if [ -n "$p" ]; then kill $p && echo "stopped $1 ($p)"; fi
    for _ in $(seq 60); do [ -z "$(sup_of "$1")$(pid_of "$2" "$3")" ] && return 0; sleep 0.5; done
    p=$(sup_of "$1")
    if [ -n "$p" ]; then kill $p; fi
}

case "${1:-start}" in
    start) start engine engine.py engine.log "$PY"; start tts tts.py tts.log "$TTS_PY"
           start accent accent.py accent.log "$PY"; start web app.py web.log "$PY"
           echo "open http://localhost:${SEOTTURI_PORT:-8787}" ;;
    stop) stop web app.py "$PY"; stop accent accent.py "$PY"; stop tts tts.py "$TTS_PY"; stop engine engine.py "$PY" ;;
    restart-web) stop web app.py "$PY"; start web app.py web.log "$PY" ;;
    restart-tts) stop tts tts.py "$TTS_PY"; start tts tts.py tts.log "$TTS_PY" ;;
    restart-accent) stop accent accent.py "$PY"; start accent accent.py accent.log "$PY" ;;
    restart-engine) stop engine engine.py "$PY"; start engine engine.py engine.log "$PY" ;;
    status) curl -s 127.0.0.1:8788/status | "$PY" -m json.tool --no-ensure-ascii | head -60
            curl -s 127.0.0.1:8789/status; echo
            curl -s 127.0.0.1:8790/status; echo ;;
    *) echo "usage: $0 {start|stop|restart-web|restart-tts|restart-accent|restart-engine|status}"; exit 1 ;;
esac

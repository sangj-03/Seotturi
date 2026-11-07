#!/usr/bin/env bash
# Keep a server running: restart it when it exits, waiting 1, 2, 4 ... up to 30 minutes between
# attempts (reset after it has run for 10 minutes), so a crash at load time cannot thrash the GPU.
#   scripts/supervise.sh <name> <command...>
name=$1
shift
delay=60
while :; do
    start=$(date +%s)
    "$@"
    code=$?
    [ -f "$(dirname "$0")/../state/stop-$name" ] && exit 0
    [ $(( $(date +%s) - start )) -gt 600 ] && delay=60
    echo "$(date '+%F %T') $name exited with $code; restarting in ${delay}s"
    sleep "$delay"
    delay=$(( delay * 2 > 1800 ? 1800 : delay * 2 ))
done

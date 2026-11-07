#!/usr/bin/env bash
# Keeps run_stage5_r64.sh going: if it dies, it is started again after 5 minutes and train.py resumes from
# the newest checkpoint. Gives up after 5 exits in a row without a new checkpoint.
#
#   cd /data/seotturi/gemma4-dialect && setsid nohup ./keep_stage5_r64.sh > /dev/null 2>&1 < /dev/null &
#   stop: kill -- -$(ps -o sid= -p $(pgrep -f '^bash \./keep_stage5_r64.sh') | tr -d " ")   (wrapper, stage script, train.py)
#   (anchor the pattern: an unanchored one also matches the shell running the command and kills it)
cd "$(dirname "$0")"
LOG=runs/stage5_r64.log
newest() { ls -d runs/lora_r64/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1; }
fails=0
while :; do
    before=$(newest)
    ./run_stage5_r64.sh >> "$LOG" 2>&1 < /dev/null
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "$(date '+%F %T') keepalive: run_stage5_r64.sh finished" >> "$LOG"
        exit 0
    fi
    after=$(newest)
    if [ "${after:-0}" -gt "${before:-0}" ]; then fails=0; else fails=$((fails + 1)); fi
    echo "$(date '+%F %T') keepalive: run_stage5_r64.sh exited with $rc at checkpoint ${after:-none}" \
         "($fails exit(s) in a row without a new checkpoint)" >> "$LOG"
    if [ "$fails" -ge 5 ]; then
        echo "$(date '+%F %T') keepalive: giving up" >> "$LOG"
        exit 1
    fi
    sleep 300
done

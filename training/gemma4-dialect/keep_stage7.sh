#!/usr/bin/env bash
# Keeps run_stage7.sh going: if it dies, it is started again after 5 minutes and train.py resumes from
# the newest checkpoint. Gives up after 5 exits in a row without a new checkpoint.
#
#   Start it from cron, not from a terminal (systemd-oomd kills a terminal's whole scope under memory pressure):
#   a one-off crontab line `M H D Mo * cd training/gemma4-dialect && setsid nohup ./keep_stage7.sh > /dev/null 2>&1 < /dev/null &`,
#   removed once it has run
#   stop: kill -- -$(ps -o sid= -p $(pgrep -f '^bash \./keep_stage7.sh') | tr -d " ")   (wrapper, stage script, train.py)
#   (anchor the pattern: an unanchored one also matches the shell running the command and kills it)
cd "$(dirname "$0")"
LOG=runs/stage7.log
newest() { ls -d runs/lora_s7/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1; }
fails=0
while :; do
    before=$(newest)
    ./run_stage7.sh >> "$LOG" 2>&1 < /dev/null
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "$(date '+%F %T') keepalive: run_stage7.sh finished" >> "$LOG"
        exit 0
    fi
    after=$(newest)
    if [ "${after:-0}" -gt "${before:-0}" ]; then fails=0; else fails=$((fails + 1)); fi
    echo "$(date '+%F %T') keepalive: run_stage7.sh exited with $rc at checkpoint ${after:-none}" \
         "($fails exit(s) in a row without a new checkpoint)" >> "$LOG"
    if [ "$fails" -ge 5 ]; then
        echo "$(date '+%F %T') keepalive: giving up" >> "$LOG"
        exit 1
    fi
    sleep 300
done

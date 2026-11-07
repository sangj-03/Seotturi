#!/bin/bash
# Stops the character drawing (gen.py) before memory gets so tight that systemd-oomd or the kernel would
# stop something else (the site's engine shares this memory): available < 6 GB or pressure avg10 > 40 %.
while P=$(pgrep -f 'python -u gen.py (encode|draw)' | head -1); [ -n "$P" ]; do
  avail=$(free -m | awk '/Mem:/{print $7}')
  psi=$(awk '/some/{split($2,a,"="); print int(a[2])}' /proc/pressure/memory)
  if [ "$avail" -lt 6000 ] || [ "$psi" -gt 40 ]; then
    echo "$(date +%T) stopping gen.py (pid $P): available ${avail} MB, pressure ${psi}%"
    kill "$P"; exit 1
  fi
  sleep 3
done
echo "$(date +%T) gen.py finished; watchdog done"

#!/usr/bin/env bash
# Keep the video check worker running: restart it 10 s after it stops for any reason.
# From Windows (Git Bash or PowerShell):
#   wsl.exe -d Ubuntu-24.04 -e bash -lc '~/graspgrade/web/worker/run_worker.sh'
cd ~/graspgrade || exit 1
while true; do
  .venv/bin/python web/worker/worker.py
  echo "worker stopped ($?), restarting in 10 s"
  sleep 10
done

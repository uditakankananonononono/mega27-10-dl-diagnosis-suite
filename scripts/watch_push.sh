#!/bin/bash
# Crash-resilience watcher: push committed state of results/ every 2 min.
cd "$(dirname "$0")/.."
export GIT_SSH_COMMAND="ssh -i ~/.ssh/mega27_10a_key -o StrictHostKeyChecking=accept-new"
while true; do
  sleep 120
  if [ -n "$(git status --porcelain results/ run_*.py)" ]; then
    git add results/ run_*.py 2>/dev/null
    git commit -q -m "wip: results checkpoint $(date -u +%H:%M)" 2>/dev/null
    git pull -q --rebase --autostash 2>/dev/null
    git push -q origin HEAD 2>/dev/null && echo "pushed $(date -u +%H:%M)"
  fi
done

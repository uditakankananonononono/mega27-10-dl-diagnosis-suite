#!/bin/bash
set -e
cd ~/mega27-10a/repo
export GIT_SSH_COMMAND='ssh -i ~/.ssh/mega27_10a_key -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10'
git add -A
git commit -qm "${1:-checkpoint}" 2>/dev/null || true
git fetch -q origin main
git rebase --autostash origin/main
git push -q origin HEAD:main
git rev-parse --short HEAD

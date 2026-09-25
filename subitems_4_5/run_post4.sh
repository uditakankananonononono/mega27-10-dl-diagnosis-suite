#!/bin/bash
# post4: rerun the three failed post3 stages after ALL_DONE, sequential.
cd ~/mega27-10b
LOG=results/post4.log
echo "[post4] waiting for ALL_DONE $(date)" >> $LOG
until grep -q ALL_DONE results/post3.log 2>/dev/null; do sleep 30; done
echo "[post4] post3 done; starting reruns $(date)" >> $LOG
sync_repo() {
  msg="$1"
  git add -A results src
  git commit -qm "$msg" || echo "[post4] WARN commit empty: $msg" >> $LOG
  git pull --rebase --autostash -q || echo "[post4] PULL FAILED: $msg" >> $LOG
  GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" git push -q || echo "[post4] PUSH FAILED: $msg" >> $LOG
  echo "[post4] synced: $msg" >> $LOG
}
run_stage() {
  script="$1"; log="$2"; marker="$3"; msg="$4"
  echo "[post4] start $script $(date)" >> $LOG
  python3 $script >> $log 2>&1
  grep -q "$marker" $log && sync_repo "$msg" || echo "[post4] STAGE FAILED: $script $(date)" >> $LOG
  echo "[post4] end $script $(date)" >> $LOG
}
run_stage src/tool_battery_7_malaria.py results/battery7_malaria.log BATTERY7_MALARIA_DONE "10.x tool battery 7 malaria rerun (p_val fix)"
run_stage src/tool_battery_7_pneu.py results/battery7_pneu.log BATTERY7_PNEU_DONE "10.x tool battery 7 pneumonia rerun"
run_stage src/tool_battery_8_pneu.py results/battery8_pneu.log BATTERY8_PNEU_DONE "10.x tool battery 8 pneumonia rerun (skopt n_calls=10)"
echo "[post4] POST4_ALL_DONE $(date)" >> $LOG

#!/bin/bash
# Second chain: pneumonia 5-tool gap + battery 8, after run_post.sh ALL_DONE.
cd ~/mega27-10b
LOG=results/post2.log
REPO=~/mega27-10-dl-diagnosis-suite
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

sync_repo() {
  local msg="$1"
  cp -r results/. $REPO/subitems_4_5/results/ 2>/dev/null
  cp src/*.py $REPO/subitems_4_5/src/ 2>/dev/null
  cp figures/*.png figures/*.html $REPO/subitems_4_5/figures/ 2>/dev/null
  cd $REPO || return
  git add -A subitems_4_5
  git commit -qm "$msg" || echo "[post2] WARN commit empty/failed: $msg" >> $LOG
  git pull --rebase --autostash -q || echo "[post2] PULL FAILED: $msg" >> $LOG
  GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" git push -q || echo "[post2] PUSH FAILED: $msg" >> $LOG
  echo "[post2] synced: $msg" >> $LOG
  cd ~/mega27-10b
}

echo "[post2] waiting for post chain $(date)" >> $LOG
until grep -q "ALL_DONE" results/post.log 2>/dev/null; do sleep 60; done

echo "[post2] battery8 pneu start $(date)" >> $LOG
python3 src/tool_battery_8_pneu.py >> results/battery8_pneu.log 2>&1
grep -q "BATTERY8_PNEU_DONE" results/battery8_pneu.log && sync_repo "10.x tool battery 8 pneumonia (SimpleITK, grad-cam, scikit-optimize)"
echo "[post2] battery8 end $(date)" >> $LOG

echo "[post2] optuna pneu start $(date)" >> $LOG
python3 src/run_optuna_pneu.py >> results/optuna_pneu.log 2>&1
grep -q "OPTUNA_PNEU_DONE" results/optuna_pneu.log && sync_repo "10.x optuna CNN search (pneumonia)"
echo "[post2] optuna pneu end $(date)" >> $LOG

echo "[post2] ablation pneu start $(date)" >> $LOG
python3 src/run_ablation_pneu.py >> results/ablation_pneu.log 2>&1
grep -q "ABLATION_PNEU_DONE" results/ablation_pneu.log && sync_repo "10.x albumentations ablation (pneumonia)"
echo "[post2] ALL_DONE $(date)" >> $LOG

#!/bin/bash
# Post-queue chain: optuna -> ablation -> battery7 malaria -> battery7 pneu.
cd ~/mega27-10b
LOG=results/post.log
REPO=~/mega27-10-dl-diagnosis-suite
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

sync_repo() {
  local msg="$1"
  cp -r results/. $REPO/subitems_4_5/results/ 2>/dev/null
  cp src/*.py $REPO/subitems_4_5/src/ 2>/dev/null
  cp figures/*.png figures/*.html $REPO/subitems_4_5/figures/ 2>/dev/null
  cd $REPO || return
  git add -A subitems_4_5
  git commit -qm "$msg" || echo "[post] WARN commit empty/failed: $msg" >> $LOG
  git pull --rebase --autostash -q || echo "[post] PULL FAILED: $msg" >> $LOG
  GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no" git push -q || echo "[post] PUSH FAILED: $msg" >> $LOG
  echo "[post] synced: $msg" >> $LOG
  cd ~/mega27-10b
}

echo "[post] waiting for OPTUNA_DONE $(date)" >> $LOG
until grep -q "OPTUNA_DONE" results/optuna.log 2>/dev/null; do sleep 60; done
echo "[post] optuna done $(date)" >> $LOG
sync_repo "10.x optuna GCN hyperparameter search (malaria, 8x2)"

echo "[post] ablation start $(date)" >> $LOG
python3 src/run_albumentations_ablation.py >> results/ablation.log 2>&1
grep -q "ABLATION_DONE" results/ablation.log && sync_repo "10.x albumentations augmentation ablation (malaria)"
echo "[post] ablation end $(date)" >> $LOG

echo "[post] battery7 malaria start $(date)" >> $LOG
python3 src/tool_battery_7_malaria.py >> results/battery7_malaria.log 2>&1
grep -q "BATTERY7_MALARIA_DONE" results/battery7_malaria.log && sync_repo "10.x tool battery 7 malaria (timm probe, umap, pingouin, ydata-profiling, shap)"
echo "[post] battery7 malaria end $(date)" >> $LOG

echo "[post] battery7 pneu start $(date)" >> $LOG
python3 src/tool_battery_7_pneu.py >> results/battery7_pneu.log 2>&1
grep -q "BATTERY7_PNEU_DONE" results/battery7_pneu.log && sync_repo "10.x tool battery 7 pneumonia (torchxrayvision, kornia, shap, pingouin, ydata-profiling)"
echo "[post] ALL_DONE $(date)" >> $LOG

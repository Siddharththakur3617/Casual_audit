#!/usr/bin/env bash
# Self-resuming chain for the counterfactual study (docs/counterfactual_study.md).
# Every step is retried after a crash (this machine has native crashes / GPU drops) and skipped once its result
# file exists; all steps resume from their caches (runs/<run>/counterfactual/, runs/<run>/extract/).
#   bash tools/run_counterfactual_chain.sh cc_phys jepa_cc_phys    # cart-pole (computes the 50-split graph first)
#   bash tools/run_counterfactual_chain.sh ac_phys jepa_ac_phys    # Acrobot
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY=.venv/Scripts/python.exe
DATA=$1; RUN=$2
CKPT=runs/$RUN/checkpoints/checkpoint_epoch0200.pt
OUT=results/studies/counterfactual_${DATA}_${RUN}
MAX_TRIES=${MAX_TRIES:-8}
STEP_TIMEOUT=${STEP_TIMEOUT:-5400}      # seconds; a frozen process (GPU lost) is killed and retried

step() {   # step <done-file> <command...>
  local done_file=$1; shift
  local n=0
  while [ ! -s "$done_file" ]; do
    n=$((n + 1))
    if [ $n -gt $MAX_TRIES ]; then echo "[chain] GIVING UP on: $* (after $MAX_TRIES tries)"; return 1; fi
    echo "[chain] $(date +%H:%M:%S) try $n: $*"
    timeout $STEP_TIMEOUT "$@"
    local rc=$?
    [ -s "$done_file" ] && break
    echo "[chain] $(date +%H:%M:%S) exit code $rc, no result yet; retrying in 30 s"
    sleep 30
  done
  echo "[chain] done: $done_file"
}

if [ "$DATA" = "cc_phys" ]; then
  step results/jepa/${RUN}_${DATA}_diff_null_cal_fwer/epoch0200/results.json \
       $PY run_all.py --data datasets/$DATA --ckpt $CKPT --prune null_cal_fwer || exit 1
fi
step $OUT/epoch0200_k025/results.json          $PY -m studies.counterfactual --data datasets/$DATA --ckpt $CKPT || exit 1
step $OUT/epoch0200_k050/results.json          $PY -m studies.counterfactual --data datasets/$DATA --ckpt $CKPT --kappa 0.5 || exit 1
step $OUT/epoch0200_k025_rollout/results.json  $PY -m studies.counterfactual --data datasets/$DATA --ckpt $CKPT --rollout || exit 1
echo "[chain] ALL DONE $DATA"

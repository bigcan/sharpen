#!/bin/bash
# gmgp1-spx500 walk-forward MULTISEED supervisor — runs ON gpuhub-1, not locally.
#
# Why a remote self-draining supervisor rather than the existing launcher:
# scripts/launch_spx500_stage.py fills whatever slots are free ONCE and prints the
# remainder as "queued", expecting a human to re-invoke it when the batch drains. This
# stage is 40 runs at ~5.3h each on 4 slots (~53h of wall clock, 10 batches). A local
# loop cannot survive that — the harness reaps long-running local loops, which is exactly
# how S555 lost runs. So the queue lives on the box and drains itself.
#
# SYMMETRY IS ENFORCED STRUCTURALLY, not checked afterwards. Jobs are only ever launched
# as a matched (ls, lo) pair sharing one (fold, seed) cell, and a new pair starts only
# when at most one pair is still running. At every instant the two arms have equal counts
# and equal GPU exposure — the pair's GPU assignment flips with pair parity so neither
# variant accumulates on gpu0. S555 lost runs to duplicate/asymmetric launches; making
# asymmetry unrepresentable is cheaper than detecting it.
#
# Queue order is SEED-MAJOR (all 4 folds at seed 42, then all 4 at seed 123, ...) so that
# at any point every fold carries the same number of seeds. Fold-major ordering would give
# fold 1 five seeds while fold 4 had none, and a partially-drained queue would be
# uninterpretable for exactly the comparison this stage exists to make.
set -u

cd /workspace/DeepScalper || exit 1
export PATH=/root/miniconda3/bin:$PATH

STAMP="$1"
SEEDS="$2"          # comma separated
FOLDS="$3"          # comma separated
SUPLOG="wfms_supervisor_${STAMP}.log"

log() { echo "[$(date '+%F %T')] $*" >> "$SUPLOG"; }

# Count only THIS stage's workers. Matching on the run_name substring keeps the count
# blind to any unrelated pipeline that might be started on the box later.
running() {
  ps -eo args | grep 'python -u scripts/run_full_pipeline' | grep -v grep | grep -c 'wfms' || true
}

launch() { # variant fold seed gpu
  local v="$1" f="$2" s="$3" g="$4"
  local rn="spx500-${v}-wfms-f${f}-s${s}_${STAMP}"
  local lg="wfms_${v}_f${f}_s${s}_${STAMP}.log"
  (
    export CUDA_VISIBLE_DEVICES="$g"
    ulimit -n 65535 2>/dev/null || true
    nohup python -u scripts/run_full_pipeline.py \
      --config "configs/gmgp1_spx500_${v}_wf_f${f}.yaml" \
      --agent sac --stage wf --seed "$s" --run_name "$rn" \
      > "$lg" 2>&1 &
  )
  log "LAUNCH $rn gpu$g -> $lg"
}

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
IFS=',' read -r -a FOLD_ARR <<< "$FOLDS"

log "SUPERVISOR START stamp=$STAMP seeds=$SEEDS folds=$FOLDS"
log "total pairs=$(( ${#SEED_ARR[@]} * ${#FOLD_ARR[@]} ))  total runs=$(( ${#SEED_ARR[@]} * ${#FOLD_ARR[@]} * 2 ))"

pair=0
for s in "${SEED_ARR[@]}"; do
  for f in "${FOLD_ARR[@]}"; do
    # Admit a new pair only with room for BOTH halves: 4 slots, so wait for <=2 busy.
    while [ "$(running)" -gt 2 ]; do sleep 60; done
    if [ $(( pair % 2 )) -eq 0 ]; then gls=0; glo=1; else gls=1; glo=0; fi
    launch ls "$f" "$s" "$gls"
    sleep 20
    launch lo "$f" "$s" "$glo"
    # Give both workers time to register in ps before the next occupancy read; a
    # too-early snapshot is what caused a legitimate run to be killed in S555.
    sleep 90
    log "PAIR $pair (f$f s$s) launched; running=$(running)"
    pair=$(( pair + 1 ))
  done
done

log "QUEUE DRAINED (${pair} pairs launched); waiting on stragglers"
while [ "$(running)" -gt 0 ]; do sleep 120; done
log "ALL RUNS COMPLETE"

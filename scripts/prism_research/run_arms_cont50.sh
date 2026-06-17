#!/usr/bin/env bash
# ============================================================================
# CONT-50 PRISM daily 3-arm re-eval runner (BTC / gold / EURUSD)
#   arm_base : zero-shot amazon/chronos-2  (NEVER actually tested before cont-50)
#   arm_v2   : existing chronos_finetuned_v2 (reuses cont-49 features verbatim)
#   arm_ft3  : fresh per-asset LoRA fine-tune, leak-safe (train <= 2022-12-31)
#
# Headline = clean OOS window (2023-01-01 -> end): arm_base + arm_ft3 never trained
# on it; arm_v2 was fine-tuned through ~2026 so its OOS is a leaky reference only.
#
# Idempotent + resumable: skips any step whose output already exists. CPU-only
# (SAFFS venv has torch+cpu) — expect a multi-hour run. Logs to run_log_cont50.txt.
# Usage:  bash scripts/prism_research/run_arms_cont50.sh
# ============================================================================
set -euo pipefail
export MSYS_NO_PATHCONV=1            # stop Git-Bash mangling C:/... arg paths
export PRISM_USE_DB=false            # in-process provider never uses the DB

cd "$(dirname "$0")/../.."           # -> project root

SAFFS_PY="${SAFFS_PY:-C:/FinRL/SAFFS/.venv/Scripts/python.exe}"   # has chronos/torch/hmmlearn
PROJ_PY="${PROJ_PY:-C:/FinRL/FinRL-Pro_DS/.venv/Scripts/python.exe}"  # has scipy (eval/compare)
PARQUET="${OHLCV_PARQUET:-data/prism_research/daily_ohlcv_long_repaired.parquet}"  # cont-49 clean data
V2_PATH="${V2_PATH:-C:/FinRL/SAFFS/models/chronos_finetuned_v2}"
ASSETS="btc gold eurusd"
OOS_START="2023-01-01"
TRAIN_END="2022-12-31"
RES=results/prism_research
LOG="$RES/run_log_cont50.txt"
mkdir -p "$RES/arm_base" "$RES/arm_v2" "$RES/arm_ft3" "$RES/models"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "================ CONT-50 START (CPU; data=$PARQUET) ================"

# ---- arm_v2: reuse cont-49 features (default model was chronos_finetuned_v2) ----
for a in $ASSETS; do
  src="$RES/prism_features_${a}_daily.parquet"
  dst="$RES/arm_v2/prism_features_${a}_daily.parquet"
  if [[ -f "$src" && ! -f "$dst" ]]; then cp "$src" "$dst"; log "arm_v2: reused cont-49 features for $a"; fi
done

# ---- arm C: fresh per-asset LoRA fine-tunes (leak-safe) ----
for a in $ASSETS; do
  out="$RES/models/chronos_ft_pathA_${a}"
  if [[ -f "$out/model.safetensors" ]]; then log "ft: $a model exists -> skip"; continue; fi
  log "ft: fine-tuning $a (1000 steps, train<=$TRAIN_END, CPU) ..."
  "$SAFFS_PY" scripts/prism_research/finetune_chronos_pathA.py \
    --ticker "$a" --train-end "$TRAIN_END" --ohlcv-parquet "$PARQUET" \
    --num-steps 1000 --batch-size 16 --device cpu --output "$out" >>"$LOG" 2>&1
  log "ft: $a done"
done

# ---- arm_base: precompute with zero-shot base model ----
need_base=0; for a in $ASSETS; do [[ -f "$RES/arm_base/prism_features_${a}_daily.parquet" ]] || need_base=1; done
if [[ $need_base -eq 1 ]]; then
  log "base: precompute (zero-shot amazon/chronos-2) for $ASSETS ..."
  env PRISM_CHRONOS_FINETUNED=__BASE_ZEROSHOT__ "$SAFFS_PY" \
    scripts/prism_research/precompute_prism_pathA.py \
    --assets $ASSETS --ohlcv-parquet "$PARQUET" --output-dir "$RES/arm_base" \
    --repair-ohlc --chronos-device cpu >>"$LOG" 2>&1
  log "base: done"
else
  log "base: all features exist -> skip"
fi

# ---- arm_ft3: precompute per-asset with each fine-tuned model ----
for a in $ASSETS; do
  dst="$RES/arm_ft3/prism_features_${a}_daily.parquet"
  if [[ -f "$dst" ]]; then log "ft3: $a features exist -> skip"; continue; fi
  log "ft3: precompute $a (fine-tuned chronos_ft_pathA_$a) ..."
  env PRISM_CHRONOS_FINETUNED="$RES/models/chronos_ft_pathA_${a}" "$SAFFS_PY" \
    scripts/prism_research/precompute_prism_pathA.py \
    --assets "$a" --ohlcv-parquet "$PARQUET" --output-dir "$RES/arm_ft3" \
    --repair-ohlc --chronos-device cpu >>"$LOG" 2>&1
  log "ft3: $a done"
done

# ---- evals: full history + clean OOS, per arm (project venv has scipy) ----
for arm in base v2 ft3; do
  log "eval: arm_$arm full-history ..."
  "$PROJ_PY" scripts/prism_research/prism_predictive_eval.py \
    --results-dir "$RES/arm_${arm}" --arm-label "$arm" --assets $ASSETS >>"$LOG" 2>&1
  log "eval: arm_$arm OOS ($OOS_START -> end) ..."
  "$PROJ_PY" scripts/prism_research/prism_predictive_eval.py \
    --results-dir "$RES/arm_${arm}" --arm-label "$arm" --oos-start "$OOS_START" --assets $ASSETS >>"$LOG" 2>&1
done

# ---- cross-arm comparison (5bps) ----
# NOTE: half-cost (2.5bps) robustness is run CONDITIONALLY (only on a flagged OOS edge,
# into a scratch dir) so it never clobbers the 5bps _oos.csv that compare_arms reads.
log "compare: cross-arm (5bps OOS headline) ..."
"$PROJ_PY" scripts/prism_research/compare_arms.py --arms base v2 ft3 --assets $ASSETS >>"$LOG" 2>&1

log "================ CONT-50 DONE ================"

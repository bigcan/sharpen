#!/usr/bin/env bash
# Run FinRL-X's Adaptive Rotation backtest through its own, unmodified deploy.sh and archive what it used and produced
# under results/finrl_x/repro/<label>/ (FinRL-X stdout, weights, summary, chart, exact price files, provenance).
#
# usage: scripts/research/finrl_x_run_rotation.sh <label> <start> <end> <priceonly|adjusted>
#   priceonly  deploy.sh as shipped: Yahoo auto_adjust=False (price-only closes)
#   adjusted   deploy.sh's own downloader with only auto_adjust flipped to True (dividend-adjusted closes)
# env: FINRLX_ROOT  FinRL-Trading clone with its own .venv (default /c/FinRL/FinRL-Trading)
#
# Then re-derive and parity-check: scripts/research/finrl_x_rotation_repro.py (see docs/research/finrl_x_reproduction_2026-09-24.md)
set -uo pipefail

[ $# -eq 4 ] || { echo "usage: $0 <label> <start> <end> <priceonly|adjusted>" >&2; exit 2; }
label=$1; start=$2; end=$3; basis=$4
root=${FINRLX_ROOT:-/c/FinRL/FinRL-Trading}
repo=$(cd "$(dirname "$0")/../.." && pwd)
py=$root/.venv/Scripts/python.exe
cfg=src/strategies/AdaptiveRotationConf_v1.2.1.yaml
out=$repo/results/finrl_x/repro/$label
[ -x "$py" ] || { echo "no venv python at $py" >&2; exit 2; }

# deploy.sh calls bare python3/pip3; route both to the clone's venv
shim=$(mktemp -d); dl=""
trap 'rm -rf "$shim"; [ -n "$dl" ] && rm -f "$dl"' EXIT
printf '#!/bin/sh\nexec "%s" "$@"\n' "$py" > "$shim/python3"
printf '#!/bin/sh\nexec "%s" -m pip "$@"\n' "$py" > "$shim/pip3"
chmod +x "$shim/python3" "$shim/pip3"
# PYTHONUTF8: their docs assume macOS; on Windows the YAML would otherwise be read as the locale code page
export PATH="$shim:$PATH" PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

case $basis in
  priceonly)
    ddir=data/fmp_daily; extra=() ;;
  adjusted)
    ddir=data/fmp_daily_adj; extra=(--skip-download)
    # First <<'PYEOF' block of deploy.sh is its downloader. Flip auto_adjust and nothing else.
    dl=$(mktemp --suffix=.py)
    awk "/<<'PYEOF'\$/{f=1; next} /^PYEOF\$/{if (f) exit} f" "$root/deploy.sh" > "$dl"
    n=$(grep -c "auto_adjust=False" "$dl")
    [ "$n" -eq 1 ] || { echo "expected exactly one auto_adjust=False in deploy.sh downloader, found $n" >&2; exit 1; }
    sed -i 's/auto_adjust=False/auto_adjust=True/' "$dl"
    # mkdir first: if the dir is missing, deploy.sh ignores --skip-download and silently refills it price-only
    mkdir -p "$root/$ddir"
    (cd "$root" && "$py" "$dl" "$cfg" "$ddir") || { echo "adjusted download failed" >&2; exit 1; }
    ;;
  *) echo "basis must be priceonly or adjusted" >&2; exit 2 ;;
esac

mkdir -p "$out"
t0=$(date +%s)
(cd "$root" && ./deploy.sh --strategy adaptive_rotation --mode backtest --start "$start" --end "$end" \
  --data-dir "$ddir" "${extra[@]}") > "$out/finrlx_stdout.log" 2>&1
rc=$?

wd=$root/src/strategies/output/weights/adaptive_rotation
for f in "ars_portfolio_weights_${start}_to_${end}.csv" "backtest_${start}_to_${end}.csv" "backtest_${start}_to_${end}.png"; do
  cp "$wd/$f" "$out/" 2>/dev/null || echo "missing output: $f" >&2
done
mkdir -p "$out/prices" && cp "$root/$ddir"/*.csv "$out/prices/"
{
  echo "clone_commit=$(git -C "$root" rev-parse HEAD)"
  echo "clone_dirty=$(git -C "$root" status --porcelain | wc -l)"
  echo "python=$("$py" --version 2>&1)"
  "$py" -m pip freeze 2>/dev/null | grep -iE '^(pandas|numpy|scipy|yfinance|pandas.market.calendars|pydantic|PyYAML)=='
  echo "basis=$basis args=--start $start --end $end --data-dir $ddir ${extra[*]}"
  echo "deploy_rc=$rc runtime_s=$(( $(date +%s) - t0 ))"
} > "$out/provenance.txt"
echo "[$label] deploy.sh rc=$rc in $(( $(date +%s) - t0 ))s -> $out"
exit $rc

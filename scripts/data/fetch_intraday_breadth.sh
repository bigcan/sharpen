#!/usr/bin/env bash
# Fetch the instrument breadth Crucible's cross-sectional leg needs at intraday depth.
#
# WHY. The extended xsec sweep (2026-08-02) showed the binding cross-sectional MDE falls to 0.394 at
# holdout 27,714 — an ALLOW. But the measured breadth grid floor is n=12 and the consumer picks the
# largest measured n NOT EXCEEDING the substrate's, so a 1- or 5-instrument panel cannot be stamped
# at all. This fetches enough instruments to clear that floor with margin.
#
# Already on disk (5): EURUSD GBPUSD USDJPY USDCHF AUDUSD, plus XAUUSD.
# This adds 9 more -> 15 total, comfortably over the n=12 floor.
#
# YEAR-BY-YEAR, because the fetcher accumulates all ticks in memory before writing: a single
# multi-year pull OOM'd at 188M ticks on 2026-08-01.
#
# PARALLEL IS ONLY SAFE BECAUSE OF baec980e. Before that fix a throttled HTTP response was booked as
# "market closed", and running blocks concurrently silently cost ~43% of each block's first year with
# ZERO failures logged. Workers are kept at 6 per stream to stay well under the throttle.
set -u
cd "$(dirname "$0")/../.."

fetch_one() {
  local instr="$1"
  for y in $(seq 2008 2026); do
    local ne=$((y + 1)); local end="${ne}-01-01"
    [ "$y" -eq 2026 ] && end="2026-08-01"
    python scripts/data/fetch_dukascopy.py --instrument "$instr" \
      --start "${y}-01-01" --end "$end" --bar 1h --workers 6 --out data/dukascopy \
      2>&1 | grep -E "wrote|FAILED|REFUS" || true
  done
  echo "=== $instr DONE ==="
}

# 4 parallel streams; grouped so each stream carries a similar total
( fetch_one USDCAD;        fetch_one NZDUSD;        fetch_one EURJPY ) &
( fetch_one EURGBP;        fetch_one XAGUSD ) &
( fetch_one USA500IDXUSD;  fetch_one USATECHIDXUSD ) &
( fetch_one DEUIDXEUR;     fetch_one LIGHTCMDUSD ) &
wait
echo "=== ALL BREADTH INSTRUMENTS DONE ==="

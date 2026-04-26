#!/usr/bin/env bash
# E1 Step 0b — Capture-mode deploy runbook
# Pairs with PR #5 (fix/e1-step-0b-capture-flag → April2026).
#
# Purpose: at the 2026-04-27 00:00 UTC un-halt, rebuild the live-engine image
# with the capture-callback hook and force-recreate gmgp1-btc + sg1-btc with
# capture mode active in their YAMLs. Capture runs for 24h or until the next
# position_mismatch halt (whichever first), producing the Tier-3 live-replay
# fixture for the E1 signed-lots refactor.
#
# Phases (each idempotent, safe to re-run):
#   preflight  — verify PR merged + YAMLs flipped + docker context alive
#   build      — rebuild finrl-live-engine image on desktop
#   deploy     — force-recreate ONLY gmgp1-btc + sg1-btc (other strategies untouched)
#   verify     — wait for /app/state/ccxt_capture_*.jsonl to appear
#   pull       — copy capture fixtures from containers to tests/fixtures/live_replay/
#   status     — show current capture state (no side effects)
#
# Usage:
#   ./scripts/runbook_e1_step0b_capture.sh preflight
#   ./scripts/runbook_e1_step0b_capture.sh build
#   ./scripts/runbook_e1_step0b_capture.sh deploy
#   ./scripts/runbook_e1_step0b_capture.sh verify
#   ./scripts/runbook_e1_step0b_capture.sh pull        # after 24h or next halt
#   ./scripts/runbook_e1_step0b_capture.sh all         # preflight + build + deploy
#
# Run from your dev laptop. Uses docker --context finrl-desktop for remote
# execution.

set -euo pipefail

DOCKER_CTX="finrl-desktop"
CONTAINERS=("gmgp1-btc" "sg1-btc")
EXPECTED_COMMIT_PREFIX="0662f919"   # broker hook commit (PR #5 commit 1/2)
EXPECTED_CONFIG_COMMIT="5fd67621"   # YAML flip commit (PR #5 commit 2/2)
FIXTURE_DIR="tests/fixtures/live_replay"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
red() { printf '\033[31m%s\033[0m\n' "$*"; }
yellow() { printf '\033[33m%s\033[0m\n' "$*"; }

fail() { red "FAIL: $*"; exit 1; }

phase_preflight() {
    bold "==> PREFLIGHT"

    # 1. Docker context reachable
    if ! docker --context "$DOCKER_CTX" info --format "{{.ServerVersion}}" >/dev/null 2>&1; then
        fail "Docker context '$DOCKER_CTX' unreachable. Check desktop power/SSH."
    fi
    green "  [ok] docker context $DOCKER_CTX reachable"

    # 2. PR commits on origin/April2026
    git -C "$REPO_ROOT" fetch origin April2026 --quiet 2>/dev/null || \
        yellow "  [warn] git fetch failed (offline?) — skipping PR-merge check"
    if git -C "$REPO_ROOT" log origin/April2026 --oneline 2>/dev/null | grep -q "$EXPECTED_COMMIT_PREFIX"; then
        green "  [ok] commit $EXPECTED_COMMIT_PREFIX on origin/April2026"
    else
        fail "commit $EXPECTED_COMMIT_PREFIX NOT on origin/April2026 — PR #5 not merged yet"
    fi
    if git -C "$REPO_ROOT" log origin/April2026 --oneline 2>/dev/null | grep -q "$EXPECTED_CONFIG_COMMIT"; then
        green "  [ok] commit $EXPECTED_CONFIG_COMMIT on origin/April2026"
    else
        fail "commit $EXPECTED_CONFIG_COMMIT NOT on origin/April2026 — config flip not merged"
    fi

    # 3. YAMLs have the capture flag
    for f in configs/live_gmgp1_btc_bybit.yaml configs/live_sg1_btc_binance.yaml; do
        if grep -q "capture_ccxt_raw_responses: true" "$REPO_ROOT/$f"; then
            green "  [ok] $f has capture flag"
        else
            fail "$f missing 'capture_ccxt_raw_responses: true'"
        fi
    done

    # 4. Container halt status (informational)
    bold "  Container status:"
    for c in "${CONTAINERS[@]}"; do
        local phase
        phase=$(docker --context "$DOCKER_CTX" exec "$c" sh -c \
            'cat /tmp/health_status.json 2>/dev/null' 2>/dev/null \
            | python -c "import json,sys;print(json.load(sys.stdin).get('phase','?'))" 2>/dev/null \
            || echo "?")
        printf "    %s: phase=%s\n" "$c" "$phase"
    done

    green "PREFLIGHT: OK"
}

phase_build() {
    bold "==> BUILD finrl-live-engine"
    yellow "  Note: build context is the LOCAL repo. Ensure your local clone has PR #5 merged."
    yellow "  If the desktop has a separate clone, push there instead."
    "$SCRIPT_DIR/manage_strategies.sh" build finrl-live-engine
    green "BUILD: OK"
}

phase_deploy() {
    bold "==> DEPLOY (force-recreate ${CONTAINERS[*]} ONLY)"
    bold "  Other strategies will NOT be touched."

    # Use compose directly to get --force-recreate (manage_strategies.sh up
    # doesn't forward the flag — see memory project_xauusd_crash_storm_s491).
    for c in "${CONTAINERS[@]}"; do
        printf "  - recreating %s ... " "$c"
        DOCKER_CONTEXT="$DOCKER_CTX" docker compose \
            -f "$REPO_ROOT/docker/live/docker-compose.yaml" \
            -f "$REPO_ROOT/docker/live/docker-compose.desktop.yaml" \
            up -d --force-recreate "$c" >/dev/null 2>&1 \
            && green "ok" || { red "failed"; exit 1; }
    done

    green "DEPLOY: OK"
    yellow "  Reminder: containers will resume halt-sleep on startup if risk_state.json"
    yellow "  is still in the future. Trading auto-resumes at halted_until (= 2026-04-27"
    yellow "  00:00 UTC for the current cycle). Capture starts on the first bar after that."
}

phase_verify() {
    bold "==> VERIFY capture file appears (timeout 90 min per container)"
    local deadline_each=5400  # 90 min
    for c in "${CONTAINERS[@]}"; do
        printf "  - polling %s ... " "$c"
        local end=$(($(date +%s) + deadline_each))
        local found=0
        while [ "$(date +%s)" -lt $end ]; do
            if docker --context "$DOCKER_CTX" exec "$c" sh -c \
                'ls /app/state/ccxt_capture_*.jsonl 2>/dev/null | head -1' 2>/dev/null \
                | grep -q ccxt_capture; then
                found=1
                break
            fi
            sleep 60
        done
        if [ $found -eq 1 ]; then
            green "ok"
        else
            yellow "timeout (90 min) — capture not yet active"
            yellow "    likely cause: container still halt-sleeping (un-halt at 00:00 UTC)"
        fi
    done
    green "VERIFY: complete"
}

phase_pull() {
    bold "==> PULL capture fixtures to $FIXTURE_DIR"
    mkdir -p "$REPO_ROOT/$FIXTURE_DIR"
    for c in "${CONTAINERS[@]}"; do
        local src
        src=$(docker --context "$DOCKER_CTX" exec "$c" sh -c \
            'ls -t /app/state/ccxt_capture_*.jsonl 2>/dev/null | head -1' 2>/dev/null || echo "")
        if [ -z "$src" ]; then
            yellow "  $c: no capture file (skip)"
            continue
        fi
        local dest="$REPO_ROOT/$FIXTURE_DIR/$(basename "$src")"
        printf "  - copying %s → %s ... " "$c" "$(basename "$src")"
        if docker --context "$DOCKER_CTX" cp "$c:$src" "$dest" >/dev/null 2>&1; then
            local sz
            sz=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null || echo "?")
            green "ok (${sz} bytes)"
        else
            red "failed"
        fi
    done
    green "PULL: complete"
    yellow "  Next: feed these into tests/live/test_replay_paper_session.py once the"
    yellow "  E1 signed-lots refactor (Steps 1–6) starts. They are the gold-standard"
    yellow "  Tier-3 regression test for PR #6."
}

phase_status() {
    bold "==> STATUS"
    for c in "${CONTAINERS[@]}"; do
        bold "  $c:"
        local phase health_ts capture_files
        phase=$(docker --context "$DOCKER_CTX" exec "$c" sh -c \
            'cat /tmp/health_status.json 2>/dev/null' 2>/dev/null \
            | python -c "import json,sys;d=json.load(sys.stdin);print(d.get('phase','?'),d.get('bar_count','?'))" \
            2>/dev/null || echo "? ?")
        printf "    phase=%s\n" "$phase"
        capture_files=$(docker --context "$DOCKER_CTX" exec "$c" sh -c \
            'ls -lh /app/state/ccxt_capture_*.jsonl 2>/dev/null' 2>/dev/null || echo "")
        if [ -n "$capture_files" ]; then
            green "    capture files:"
            echo "$capture_files" | sed 's/^/      /'
        else
            yellow "    no capture files yet"
        fi
        # Halt state
        local halt
        halt=$(docker --context "$DOCKER_CTX" exec "$c" sh -c \
            'cat /app/state/risk_state.json 2>/dev/null' 2>/dev/null \
            | python -c "import json,sys;d=json.load(sys.stdin);print(d.get('halted_until','none'),d.get('reason','none'))" \
            2>/dev/null || echo "none none")
        printf "    halt: until=%s reason=%s\n" $halt
    done
}

usage() {
    cat <<EOF
Usage: $0 {preflight|build|deploy|verify|pull|status|all}

Phases:
  preflight  Verify PR merged, YAMLs flipped, docker context alive
  build      Rebuild finrl-live-engine image
  deploy     Force-recreate gmgp1-btc + sg1-btc ONLY (idempotent)
  verify     Wait up to 90min/container for capture file to appear
  pull       Copy capture fixtures to tests/fixtures/live_replay/
  status     Show current capture + halt state (no side effects)
  all        preflight + build + deploy (skip verify/pull — those run later)
EOF
}

case "${1:-help}" in
    preflight) phase_preflight ;;
    build) phase_build ;;
    deploy) phase_preflight && phase_deploy ;;
    verify) phase_verify ;;
    pull) phase_pull ;;
    status) phase_status ;;
    all) phase_preflight && phase_build && phase_deploy ;;
    help|-h|--help) usage ;;
    *) usage; exit 1 ;;
esac

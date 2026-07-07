<#
.SYNOPSIS
    Manual-trigger convenience wrapper for one real Crucible Taiwan orchestrator tick.

.DESCRIPTION
    No Task Scheduler, no fixed cadence -- run this whenever you choose to check for new
    TWSE/TAIFEX data and mine it. Equivalent to:

        python scripts/research/crucible_orchestrator.py --config configs/taiwan_signal_eval.gates.yaml `
            --mode real --force --nights 1 --max-proposals 128 `
            --out results/crucible_orchestrator/taiwan_v2

    Output (ledger, FDR budget, lockbox, cards, manifests) accumulates across runs in
    results/crucible_orchestrator/taiwan_v2/ -- always point future manual runs at the same
    directory so online-FDR wealth and dedup history stay continuous.

    SUBSTRATE UPGRADE (S553-cont-117, P1): this wrapper now targets taiwan_v2, NOT the
    original taiwan_manual. taiwan_manual (v1) scored the overlays at ~267 bars (~9.5%
    coverage) -- statistically power-starved, so its 0 PROMISING was correct-at-its-power but
    uninformative. The P0 T86 backfill deepened the TWSE-institutional slots to ~2782 bars, a
    materially different data surface (new data_snapshot_hash). Because the ledger dedup key is
    the canonical formula only (no substrate component), re-running against taiwan_manual would
    hash-block the already-scored overlays; a fresh out-dir gives a fresh ledger + a fresh
    LORD++ FDR account that legitimately re-opens ALL of them at full power. taiwan_manual is
    kept, frozen, as the archived v1 record -- report BOTH, never cherry-pick. See
    docs/research/crucible_taiwan_v2_substrate_upgrade_2026-07-06.md for the pre-registration.

    --max-proposals 128 (P2): the library seed bank emits ~101 proposals (8 cross-sectional +
    3 templates x ~31 feature slots); the old default 32 truncated ~69 overlay slots
    (semi/gold/wti/esg...) that were never scored. 128 exhausts the batch so every slot enters
    the funnel at full post-backfill power.

    Running on a day with no new TWSE/TAIFEX data and no fresh hypotheses is expected to
    no-op (mined=false) -- that is substrate_dirty conserving FDR wealth correctly, not a
    failure. See .agent/artifacts/crucible_taiwan_breadth_and_scheduling_architecture.md
    (Part B) for the full unattended-scheduling design this replaces for now.

.PARAMETER Llm
    Opt-in P3: use the live LLM-backed proposer (spec Part A2) instead of the offline library
    seed bank. Requires ANTHROPIC_API_KEY in the environment; the orchestrator fails closed
    (aborts before the panel load) without it. Off by default -- the library proposer is free,
    offline, and bit-reproducible.

.PARAMETER MaxProposals
    Per-tick proposal cap forwarded as --max-proposals (default 128, the P2 value that exhausts
    the library batch). Lower it to sub-sample the batch.

.PARAMETER ExtraArgs
    Any additional flags to pass through to crucible_orchestrator.py, e.g.
    -ExtraArgs "--no-altdata-slots" or "--est-tokens-per-tick","16000".

.EXAMPLE
    .\scripts\research\taiwan_tick.ps1

.EXAMPLE
    .\scripts\research\taiwan_tick.ps1 -Llm          # LLM proposer (needs ANTHROPIC_API_KEY)

.EXAMPLE
    .\scripts\research\taiwan_tick.ps1 -ExtraArgs "--no-lockbox"
#>
param(
    [switch]$Llm,
    [int]$MaxProposals = 128,
    [string[]]$ExtraArgs = @()
)

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "logs\crucible_taiwan"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "tick_$Stamp.log"

Write-Host "Crucible Taiwan tick starting -- logging to $LogFile"

# P2 (--max-proposals) + P3 (opt-in --proposer llm) assembled here so an operator override in
# -ExtraArgs still wins (argparse takes the last occurrence of a repeated flag).
$ProposerArgs = @("--max-proposals", "$MaxProposals")
if ($Llm) { $ProposerArgs += @("--proposer", "llm") }

& python scripts/research/crucible_orchestrator.py `
    --config configs/taiwan_signal_eval.gates.yaml `
    --mode real --force --nights 1 `
    --out results/crucible_orchestrator/taiwan_v2 `
    @ProposerArgs @ExtraArgs 2>&1 | Tee-Object -FilePath $LogFile

$ExitCode = $LASTEXITCODE
$SummaryPath = "results\crucible_orchestrator\taiwan_v2\real\orchestrator_summary.json"
if ($ExitCode -eq 0) {
    Write-Host "`nDone. Summary: $SummaryPath"
    if (Test-Path $SummaryPath) {
        $Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
        $Tick = $Summary.ticks[0]
        Write-Host "  dirty=$($Tick.dirty)  mined=$($Tick.mined)  promising=$($Tick.n_promising)  reason=$($Tick.reason)"
    }
} else {
    Write-Host "`nFAILED (exit $ExitCode) -- see $LogFile" -ForegroundColor Red
}
exit $ExitCode

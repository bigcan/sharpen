<#
.SYNOPSIS
    Manual-trigger convenience wrapper for one real Crucible Taiwan orchestrator tick.

.DESCRIPTION
    No Task Scheduler, no fixed cadence -- run this whenever you choose to check for new
    TWSE/TAIFEX data and mine it. Equivalent to:

        python scripts/research/crucible_orchestrator.py --config configs/taiwan_signal_eval.gates.yaml `
            --mode real --force --nights 1 --out results/crucible_orchestrator/taiwan_manual

    Output (ledger, FDR budget, lockbox, cards, manifests) accumulates across runs in
    results/crucible_orchestrator/taiwan_manual/ -- always point future manual runs at the
    same directory so online-FDR wealth and dedup history stay continuous (S553-cont-115:
    this directory carries forward the state from that session's live pre-flight run).

    Running on a day with no new TWSE/TAIFEX data and no fresh hypotheses is expected to
    no-op (mined=false) -- that is substrate_dirty conserving FDR wealth correctly, not a
    failure. See .agent/artifacts/crucible_taiwan_breadth_and_scheduling_architecture.md
    (Part B) for the full unattended-scheduling design this replaces for now.

.PARAMETER ExtraArgs
    Any additional flags to pass through to crucible_orchestrator.py, e.g.
    -ExtraArgs "--no-altdata-slots" or "--max-proposals","16".

.EXAMPLE
    .\scripts\research\taiwan_tick.ps1

.EXAMPLE
    .\scripts\research\taiwan_tick.ps1 -ExtraArgs "--no-lockbox"
#>
param(
    [string[]]$ExtraArgs = @()
)

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "logs\crucible_taiwan"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "tick_$Stamp.log"

Write-Host "Crucible Taiwan tick starting -- logging to $LogFile"

& python scripts/research/crucible_orchestrator.py `
    --config configs/taiwan_signal_eval.gates.yaml `
    --mode real --force --nights 1 `
    --out results/crucible_orchestrator/taiwan_manual `
    @ExtraArgs 2>&1 | Tee-Object -FilePath $LogFile

$ExitCode = $LASTEXITCODE
$SummaryPath = "results\crucible_orchestrator\taiwan_manual\real\orchestrator_summary.json"
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

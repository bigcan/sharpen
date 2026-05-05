# Phase-2 off-host pull of agent-memory LanceDB backups.
#
# Phase 1 (sidecar on finrl-desktop) writes nightly tarballs into the
# agent-memory-backups Docker volume. Phase 2 (this script) copies any
# new tarballs from that volume to a local folder on this Windows host
# so a finrl-desktop disk failure doesn't take the backups with it.
#
# Uses the existing finrl-desktop Docker context — no SSH or extra creds.
#
# Usage:
#   pwsh scripts/agent_memory_backup_pull.ps1
#   pwsh scripts/agent_memory_backup_pull.ps1 -DestDir <path> -RetentionDays 30
#   pwsh scripts/agent_memory_backup_pull.ps1 -DryRun
#
# Schedule (operator-driven, not installed by default):
#   schtasks /Create /TN "FinRL-AgentMemoryBackupPull" /SC DAILY /ST 12:00 `
#     /TR "pwsh -NoProfile -File C:\FinRL\FinRL-Pro_DS\scripts\agent_memory_backup_pull.ps1" /F

[CmdletBinding()]
param(
    [string]$DestDir = (Join-Path $PSScriptRoot "..\.backups\agent-memory"),
    [string]$Container = "agent-memory-backup",
    [string]$Context = "finrl-desktop",
    [int]$RetentionDays = 30,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$DestDir = (Resolve-Path -LiteralPath $DestDir -ErrorAction SilentlyContinue) ?? $DestDir
if (-not (Test-Path -LiteralPath $DestDir)) {
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
}
$DestDir = (Resolve-Path -LiteralPath $DestDir).Path

function Log($msg) {
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Write-Host "[$ts] [agent-memory-pull] $msg"
}

Log "dest=$DestDir container=$Container context=$Context retention=${RetentionDays}d dry_run=$DryRun"

# 1. List remote tarballs
$remoteRaw = & docker --context $Context exec $Container sh -c "ls -1 /backups 2>/dev/null | grep -E '^agent-memory-.*\.tar\.gz$'" 2>&1
if ($LASTEXITCODE -ne 0) {
    Log "FATAL remote ls failed (exit=$LASTEXITCODE): $remoteRaw"
    exit 1
}
$remote = @($remoteRaw | Where-Object { $_ -and $_ -match '^agent-memory-.*\.tar\.gz$' })
Log "remote inventory: $($remote.Count) file(s)"

# 2. Diff against local
$local = @(Get-ChildItem -LiteralPath $DestDir -Filter 'agent-memory-*.tar.gz' -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
$missing = @($remote | Where-Object { $local -notcontains $_ })
Log "local inventory: $($local.Count) file(s) — missing: $($missing.Count)"

# 3. Fetch missing via docker cp
$fetched = 0
foreach ($name in $missing) {
    $localPath = Join-Path $DestDir $name
    if ($DryRun) {
        Log "DRY-RUN would fetch $name -> $localPath"
        continue
    }
    Log "fetch $name"
    & docker --context $Context cp "${Container}:/backups/$name" $localPath
    if ($LASTEXITCODE -ne 0) {
        Log "WARN docker cp failed for $name (exit=$LASTEXITCODE) — skipping"
        continue
    }
    $size = (Get-Item -LiteralPath $localPath).Length
    if ($size -lt 1024) {
        Log "WARN $name is suspiciously small ($size bytes) — keeping but flagging"
    }
    $fetched++
}

# 4. Local retention sweep — mtime-based, mirrors the in-container script
$cutoff = (Get-Date).AddDays(-$RetentionDays)
$pruned = 0
Get-ChildItem -LiteralPath $DestDir -Filter 'agent-memory-*.tar.gz' -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt $cutoff } |
    ForEach-Object {
        if ($DryRun) {
            Log "DRY-RUN would prune $($_.Name) (mtime=$($_.LastWriteTime.ToString('o')))"
        } else {
            Remove-Item -LiteralPath $_.FullName -Force
            Log "pruned $($_.Name)"
        }
        $pruned++
    }

# 5. Inventory summary
$final = Get-ChildItem -LiteralPath $DestDir -Filter 'agent-memory-*.tar.gz' -ErrorAction SilentlyContinue
$totalSize = ($final | Measure-Object -Property Length -Sum).Sum
$totalMb = if ($totalSize) { [math]::Round($totalSize / 1MB, 2) } else { 0 }
Log "done fetched=$fetched pruned=$pruned local_total=$($final.Count) size=${totalMb}MB"

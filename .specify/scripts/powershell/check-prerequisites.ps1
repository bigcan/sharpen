<#
.SYNOPSIS
  Emit repository feature directory metadata for task generation.

.DESCRIPTION
  Identifies the active feature directory under /specs that contains plan.md.
  Returns absolute paths so downstream tooling can locate design documents.
  Supports JSON output via the -Json switch.
#>

[CmdletBinding()]
param(
    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$specsDir = Join-Path $repoRoot 'specs'

if (-not (Test-Path -LiteralPath $specsDir -PathType Container)) {
    throw "Specs directory not found at '$specsDir'."
}

$featureDirs = Get-ChildItem -LiteralPath $specsDir -Directory |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'plan.md') } |
    Sort-Object Name

if (-not $featureDirs) {
    throw "No feature directory with plan.md found beneath '$specsDir'."
}

$featureDir = $featureDirs | Select-Object -First 1
$featureDirPath = [System.IO.Path]::GetFullPath($featureDir.FullName)

$docFiles = @()
$primaryDocs = @(
    'plan.md',
    'spec.md',
    'data-model.md',
    'research.md',
    'quickstart.md'
)

foreach ($doc in $primaryDocs) {
    $candidate = Join-Path $featureDirPath $doc
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $docFiles += [System.IO.Path]::GetFullPath($candidate)
    }
}

$contractsDir = Join-Path $featureDirPath 'contracts'
if (Test-Path -LiteralPath $contractsDir -PathType Container) {
    $contracts = Get-ChildItem -LiteralPath $contractsDir -File
    foreach ($contract in $contracts) {
        $docFiles += [System.IO.Path]::GetFullPath($contract.FullName)
    }
}

$result = [ordered]@{
    FEATURE_DIR    = $featureDirPath
    AVAILABLE_DOCS = $docFiles
}

if ($Json) {
    $result | ConvertTo-Json -Depth 4
} else {
    return $result
}

param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$python = Join-Path $ProjectRoot '.venv-training\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Existing training Python environment is required.' }
$logDirectory = Join-Path $ProjectRoot 'experiments\rule_forward_v1\entry_logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$runStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')
$entryLog = Join-Path $logDirectory ($runStamp + '.log')
Push-Location -LiteralPath $ProjectRoot
try {
    & $python 'scripts\run_rule_forward.py' *> $entryLog
    $runExit = $LASTEXITCODE
    Write-Output (Get-Content -LiteralPath $entryLog)
    Write-Output ('Entry log: ' + $entryLog)
    exit $runExit
} finally { Pop-Location }

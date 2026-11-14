# One manually invoked, bounded research job. No Worker/Controller changes.
[CmdletBinding()]
param([switch]$EvaluateFinal)
$ErrorActionPreference = 'Stop'
if ($env:USERNAME -ne 'LocalCTOGPUWorker') { throw 'Use the existing LocalCTOGPUWorker standard-account PowerShell.' }
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Do not run elevated.' }
$bundleRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$manifestPath = Join-Path $bundleRoot 'FILES.json'
if ((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash -ne '__MANIFEST_SHA256__') { throw 'Bundle manifest mismatch' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
foreach ($entry in $manifest.files.PSObject.Properties) {
    $targetPath = [IO.Path]::GetFullPath((Join-Path $bundleRoot $entry.Name))
    if (-not $targetPath.StartsWith($bundleRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe bundle path' }
    if ((Get-FileHash -LiteralPath $targetPath -Algorithm SHA256).Hash -ne $entry.Value) { throw ('Bundle file mismatch: ' + $entry.Name) }
}
if (Test-Path -LiteralPath (Join-Path $bundleRoot 'results/price-v1-seed42')) { throw 'Prior run exists; do not automatically repeat or tune.' }
$downloadRoot = Join-Path $env:LOCALAPPDATA 'QuantResearch\Downloads'
New-Item -ItemType Directory -Path $downloadRoot -Force > $null
$installer = Join-Path $downloadRoot 'python-3.12.10-amd64.exe'
if (-not (Test-Path -LiteralPath $installer)) {
    Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $installer -UseBasicParsing
}
if ((Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash -ne '67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb') { throw 'Official Python installer SHA256 mismatch; do not run it.' }
$signature = Get-AuthenticodeSignature -LiteralPath $installer
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') { throw 'Python installer signature invalid' }
$runtimeRoot = Join-Path $env:LOCALAPPDATA 'QuantResearch\Runtime\Python312-PriceV1'
$runtimeExe = Join-Path $runtimeRoot 'python.exe'
if (-not (Test-Path -LiteralPath $runtimeExe)) {
    # Private user installation only; no launcher, PATH, associations or shortcuts.
    $installArguments = @('/quiet', 'InstallAllUsers=0', ('TargetDir="' + $runtimeRoot + '"'), 'Include_launcher=0', 'PrependPath=0', 'AppendPath=0', 'AssociateFiles=0', 'Shortcuts=0', 'Include_test=0', 'Include_doc=0', 'Include_tcltk=0', 'Include_pip=1')
    $installation = Start-Process -FilePath $installer -ArgumentList $installArguments -Wait -PassThru -WindowStyle Hidden
    if ($installation.ExitCode -ne 0) { throw ('Python user installation exit ' + $installation.ExitCode + '; do not elevate, reboot or change policy.') }
}
$environmentRoot = Join-Path $env:LOCALAPPDATA 'QuantResearch\Environments\price-v1-__ENV_SUFFIX__'
$python = Join-Path $environmentRoot 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    & $runtimeExe -I -m venv $environmentRoot
    if ($LASTEXITCODE -ne 0) { throw 'Private training environment creation failed.' }
}
Push-Location -LiteralPath $bundleRoot
try {
    & $python -I -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-training-lock.txt --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { throw 'Isolated training dependency install failed.' }
    & $python -I -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
    # Tests are synthetic/schema checks; they do not inspect sealed market prices.
    & $python -m pytest -q tests/test_price_audit.py tests/test_price_training.py --junitxml=worker-tests.xml
    if ($LASTEXITCODE -ne 0) { throw 'Worker affected tests failed.' }
    if ($EvaluateFinal) {
        & $python -I ops/price_gpu_job.py --evaluate-final
    } else {
        & $python -I ops/price_gpu_job.py
    }
    if ($LASTEXITCODE -ne 0) { throw 'Price GPU job failed; retain claims/logs and report the error.' }
} finally {
    Pop-Location
}

# Resume only the interrupted official Python download; preserve the original job.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
if ($env:USERNAME -ne 'LocalCTOGPUWorker') { throw 'Use the existing LocalCTOGPUWorker standard account.' }
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Do not run elevated.' }
$installer = Join-Path $env:LOCALAPPDATA 'QuantResearch\Downloads\python-3.12.10-amd64.exe'
$expectedHash = '67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb'
$expectedLength = 26964224
$launcher = Join-Path $env:USERPROFILE 'Downloads\quant-price-v1-small\Start-PriceTraining.ps1'
if ((Get-FileHash -LiteralPath $launcher -Algorithm SHA256).Hash -ne '615f8ec1b0a753bb2b26f5ce3d843271bdb8b061e52ac4b0bb851c542bdc9b6e') { throw 'Original launcher SHA256 mismatch.' }
if (-not (Test-Path -LiteralPath $installer)) { throw 'Expected interrupted file is missing; stop instead of changing the download plan.' }
$actualLength = (Get-Item -LiteralPath $installer).Length
if ($actualLength -gt $expectedLength) { throw 'Unexpected installer length; do not truncate or overwrite.' }
if ($actualLength -lt $expectedLength) {
    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
    if (-not (Test-Path -LiteralPath $curl)) { throw 'Built-in curl.exe unavailable; no alternative software will be installed.' }
    Write-Host ('Resuming ' + $actualLength + ' / ' + $expectedLength + ' bytes from python.org')
    & $curl --fail --location --continue-at - --retry 1 --retry-delay 2 --retry-max-time 240 --connect-timeout 15 --max-time 180 --output $installer 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe'
    if ($LASTEXITCODE -ne 0) { throw ('Resume failed, curl exit ' + $LASTEXITCODE + '. Partial bytes retained; no automatic training retry.') }
}
if ((Get-Item -LiteralPath $installer).Length -ne $expectedLength -or (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash -ne $expectedHash) { throw 'Installer length/SHA256 mismatch; never run it.' }
Write-Host 'Official installer download verified. Continuing the unchanged fixed research job.'
& $launcher -EvaluateFinal

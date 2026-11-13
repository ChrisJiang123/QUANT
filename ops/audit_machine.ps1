param([string]$Label = 'UNCONFIRMED_MACHINE')
# Read-only system inspection. Emits sanitized JSON to stdout; no configuration changes.
# Do not include IPs, hostnames, serials, usernames, keys, environment dumps or peer names.
$ErrorActionPreference = 'Continue'
$script:failures = @()
function Probe([string]$Name, [scriptblock]$Action) {
    try { & $Action } catch { $script:failures += $Name; $null }
}
function Present([string]$Name) { [bool](Get-Command $Name -ErrorAction SilentlyContinue) }
$os = Probe 'os' { Get-CimInstance Win32_OperatingSystem }
$system = Probe 'system' { Get-CimInstance Win32_ComputerSystem }
$cpu = Probe 'cpu' { Get-CimInstance Win32_Processor }
$gpu = Probe 'gpu' { Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion }
$commands = [ordered]@{}
foreach ($name in 'python','py','git','docker','wsl','nvidia-smi','nvcc','tailscale','ssh') { $commands[$name] = Present $name }
$versions = [ordered]@{}
if ($commands.python) { $versions.python = (& python --version 2>&1 | Out-String).Trim() }
if ($commands.git) { $versions.git = (& git --version 2>&1 | Out-String).Trim() }
if ($commands.docker) { $versions.docker = (& docker --version 2>&1 | Out-String).Trim() }
if ($commands.nvcc) { $versions.nvcc = (& nvcc --version 2>&1 | Out-String).Trim() }
$nvidia = $null
if ($commands.'nvidia-smi') {
    $nvidia = @(& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null)
    $versions.nvidia_cuda_driver_capability = ((& nvidia-smi 2>$null | Select-String 'CUDA Version:').ToString()).Trim()
}
$runtime = $null
if ($commands.python) {
    $runtimeJson = & python (Join-Path $PSScriptRoot 'audit_runtime.py')
    if ($LASTEXITCODE -eq 0) { $runtime = $runtimeJson | ConvertFrom-Json }
    else { $script:failures += 'runtime_probe_failed' }
}
$remoteServices = @(Get-Service -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'tailscale|sshd|TermService|TeamViewer|AnyDesk|RustDesk' } | Select-Object Name,@{N='Status';E={$_.Status.ToString()}})
$sshConfig = Join-Path $env:USERPROFILE '.ssh/config'
$sshInfo = @{ config_exists = (Test-Path -LiteralPath $sshConfig); known_hosts_exists = (Test-Path -LiteralPath (Join-Path $env:USERPROFILE '.ssh/known_hosts')); note = 'Key files/known-host identities not read; target authorization not established by presence.' }
$tailscale = $null
if ($commands.tailscale) {
    $ts = & tailscale status --json 2>$null | ConvertFrom-Json
    $tailscale = @{ backend_state=$ts.BackendState; peer_count=@($ts.Peer.PSObject.Properties).Count }
}
$report = [ordered]@{
    schema_version = 'machine-audit-v1'
    label = $Label
    inspected_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    read_only = $true
    os = @{ caption=$os.Caption; version=$os.Version; build=$os.BuildNumber; architecture=$os.OSArchitecture }
    hardware = @{ manufacturer=$system.Manufacturer; model=$system.Model; ram_gib=[math]::Round($system.TotalPhysicalMemory/1GB,2); cpu=@($cpu | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,VirtualizationFirmwareEnabled,SecondLevelAddressTranslationExtensions) }
    disks = @(Get-Volume | Where-Object DriveLetter | Select-Object DriveLetter,FileSystem,@{N='size_gib';E={[math]::Round($_.Size/1GB,2)}},@{N='free_gib';E={[math]::Round($_.SizeRemaining/1GB,2)}})
    video_controllers = @($gpu)
    nvidia_smi = $nvidia
    commands = $commands
    versions = $versions
    runtime = $runtime
    remote_services = $remoteServices
    ssh = $sshInfo
    tailscale = $tailscale
    failed_probes = @($script:failures)
    caveats = @('nvidia-smi CUDA Version is driver capability, not proof of CUDA Toolkit installation.', 'No networking, firewall, remote-access or driver configuration was changed.', 'PyTorch probe uses the active python only; other virtual environments are not enumerated.')
}
$report | ConvertTo-Json -Depth 10

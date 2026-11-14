param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
if ($ProjectRoot.Contains('"')) { throw 'Project path must not contain quote characters.' }
$taskService = New-Object -ComObject 'Schedule.Service'
$taskService.Connect()
$definition = $taskService.NewTask(0)
$definition.RegistrationInfo.Description = 'Public-data file research only; no bot or orders.'
$definition.Principal.UserId = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$definition.Principal.LogonType = 3
$definition.Principal.RunLevel = 0
$definition.Settings.Enabled = $true
$definition.Settings.StartWhenAvailable = $false
$definition.Settings.MultipleInstances = 2
$definition.Settings.ExecutionTimeLimit = 'PT45M'
$trigger = $definition.Triggers.Create(2)
$trigger.StartBoundary = [DateTime]::Today.AddHours(6).AddMinutes(15).ToString('yyyy-MM-ddTHH:mm:ss')
$trigger.DaysInterval = 1
$action = $definition.Actions.Create(0)
$action.Path = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
$action.Arguments = '-NoProfile -File "' + (Join-Path $ProjectRoot 'ops\run_forward_daily.ps1') + '"'
$action.WorkingDirectory = $ProjectRoot
$outputDirectory = Join-Path $ProjectRoot 'experiments\rule_forward_v1'
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$outputPath = Join-Path $outputDirectory 'schedule.xml'
$definition.XmlText | Set-Content -LiteralPath $outputPath -Encoding Unicode
Write-Output ('CONFIGURATION ONLY; task not registered: ' + $outputPath)

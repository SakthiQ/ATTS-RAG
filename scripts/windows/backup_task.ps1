<#
.SYNOPSIS
  Registers (or removes) the nightly ATTS-RAG backup as a Windows scheduled task.

.DESCRIPTION
  The task runs scripts\backup.py every night. Settings come from the .env file in -DataRoot (BACKUP_DIR,
  BACKUP_AGE_RECIPIENT, BACKUP_REMOTE, BACKUP_HEARTBEAT_URL, AGE_BIN, RCLONE_BIN); see scripts\backup.py.
  If the computer is off at that time, the task runs as soon as it is on again (StartWhenAvailable).
  It runs only while you are logged on, so it needs no stored password.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\backup_task.ps1 -DataRoot C:\path\to\ATTS-RAG
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\backup_task.ps1 -Unregister
#>
param(
    # The ATTS-RAG folder holding data\, chroma_db\, logs\ and .env
    [string]$DataRoot = (Resolve-Path "$PSScriptRoot\..\..").Path,
    # The folder holding scripts\backup.py and app\ (default: this checkout)
    [string]$CodeRoot = (Resolve-Path "$PSScriptRoot\..\..").Path,
    # Python with the project's packages (default: .venv in DataRoot)
    [string]$Python = "",
    [string]$At = "02:00",
    [string]$TaskName = "ATTS-RAG nightly backup",
    [switch]$Unregister
)
$ErrorActionPreference = "Stop"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Removed the scheduled task '$TaskName'."
    return
}

if (-not $Python) { $Python = Join-Path $DataRoot ".venv\Scripts\python.exe" }
foreach ($p in @($Python, (Join-Path $CodeRoot "scripts\backup.py"), (Join-Path $DataRoot "data"))) {
    if (-not (Test-Path $p)) { throw "Not found: $p" }
}

$script = Join-Path $CodeRoot "scripts\backup.py"
$action = New-ScheduledTaskAction -Execute $Python -Argument "`"$script`" --root `"$DataRoot`"" -WorkingDirectory $CodeRoot
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description "Encrypted, checked backup of ATTS-RAG (scripts\backup.py). Result: logs\backup_status.json." -Force | Out-Null
Write-Output "Registered '$TaskName': every day at $At (or as soon as the computer is on), backing up $DataRoot."

<#
.SYNOPSIS
  Starts Uptime Kuma (the monitoring dashboard) on this computer, or registers it to start at log-on.

.DESCRIPTION
  Uptime Kuma listens on 127.0.0.1 only, so it cannot be reached from the network. Open http://127.0.0.1:3001,
  create the admin account on first start, then add the monitors listed in docs\Operations_Runbook.md.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\uptime_kuma.ps1 -Start
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\uptime_kuma.ps1 -Register
#>
param(
    [string]$KumaDir = "D:\ATTS-RAG-tools\uptime-kuma",
    [string]$DataDir = "D:\ATTS-RAG-tools\uptime-kuma-data",
    [int]$Port = 3001,
    [string]$TaskName = "ATTS-RAG Uptime Kuma",
    [switch]$Start,
    [switch]$Register,
    [switch]$Unregister
)
$ErrorActionPreference = "Stop"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Removed the scheduled task '$TaskName'."
    return
}
$node = (Get-Command node -ErrorAction Stop).Source
if (-not (Test-Path (Join-Path $KumaDir "server\server.js"))) { throw "Uptime Kuma is not installed in $KumaDir" }
New-Item -ItemType Directory -Force $DataDir | Out-Null

if ($Register) {
    # cmd sets the settings for this one process, then starts Kuma hidden
    $cmdArgs = "/c set UPTIME_KUMA_HOST=127.0.0.1&& set UPTIME_KUMA_PORT=$Port&& set DATA_DIR=$DataDir\&& `"$node`" server\server.js"
    $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $cmdArgs -WorkingDirectory $KumaDir
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -Hidden
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
        -Description "Uptime Kuma for ATTS-RAG on http://127.0.0.1:$Port" -Force | Out-Null
    Write-Output "Registered '$TaskName': starts at log-on. Start it now with: Start-ScheduledTask -TaskName '$TaskName'"
}

if ($Start) {
    $env:UPTIME_KUMA_HOST = "127.0.0.1"
    $env:UPTIME_KUMA_PORT = "$Port"
    $env:DATA_DIR = "$DataDir\"
    Start-Process -FilePath $node -ArgumentList "server\server.js" -WorkingDirectory $KumaDir -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $DataDir "kuma.log") -RedirectStandardError (Join-Path $DataDir "kuma-error.log")
    Write-Output "Uptime Kuma is starting on http://127.0.0.1:$Port (log: $DataDir\kuma.log)."
}

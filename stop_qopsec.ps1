#!/usr/bin/env pwsh
#Requires -Version 5.1
<#
.SYNOPSIS
    Para o Q-OPSEC no Windows.
.DESCRIPTION
    - Dispara POST /stop/all no orquestrador
    - Encerra o processo do orquestrador (logs\orchestrator.pid)
    - Opcionalmente para os containers criados pelo instalador (-StopContainers)
#>
param([switch]$StopContainers)

$ErrorActionPreference = 'Stop'
$InstallDir = Split-Path -Parent $PSScriptRoot
$pidFile    = Join-Path $InstallDir 'logs\orchestrator.pid'

try {
    Invoke-RestMethod -Uri 'http://127.0.0.1:8090/stop/all' -Method Post -TimeoutSec 60 | Out-Null
    Write-Host '[OK] /stop/all disparado' -ForegroundColor Green
} catch {
    Write-Warning 'Orquestrador nao respondeu ao /stop/all (provavelmente ja esta parado)'
}

if (Test-Path $pidFile) {
    $orchPid = Get-Content $pidFile
    $proc = Get-Process -Id $orchPid -ErrorAction SilentlyContinue
    if ($proc) { Stop-Process -Id $orchPid -Force -ErrorAction SilentlyContinue; Write-Host "[OK] Orquestrador PID $orchPid encerrado" -ForegroundColor Green }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

if ($StopContainers) {
    foreach ($c in @('qopsec-postgres', 'qopsec-mongo', 'qopsec-mosquitto')) {
        docker rm -f $c 2>$null | Out-Null
    }
    Write-Host '[OK] Containers de infra parados' -ForegroundColor Green
}

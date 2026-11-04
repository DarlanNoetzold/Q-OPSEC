#!/usr/bin/env pwsh
#Requires -Version 5.1
<#
.SYNOPSIS
    Inicia o Q-OPSEC no Windows (orquestrador + servicos).
.DESCRIPTION
    - Carrega os segredos de ~\.config\qopsec\middleware.env
    - Sobe o orquestrador (qopsec_env\Scripts\python.exe orchestrator.py) na 8090
    - Aguarda os containers gerenciados (Redis/RabbitMQ/...)
    - Dispara POST /start/all para iniciar todos os servicos
#>
param([switch]$SkipServices)

$ErrorActionPreference = 'Stop'
$InstallDir  = Split-Path -Parent $PSScriptRoot
$secretsFile = Join-Path $env:USERPROFILE '.config\qopsec\middleware.env'
$venvPython  = Join-Path $InstallDir 'qopsec_env\Scripts\python.exe'
$pidFile     = Join-Path $InstallDir 'logs\orchestrator.pid'

if (-not (Test-Path $venvPython)) { Write-Error "venv nao encontrado em $venvPython. Rode o install.ps1 primeiro."; exit 1 }

# 1. segredos
$envVars = @{}
if (Test-Path $secretsFile) {
    Get-Content $secretsFile | ForEach-Object {
        if ($_ -match '^([A-Za-z0-9_]+)=(.*)$') { $envVars[$matches[1]] = $matches[2] }
    }
    if ($envVars['QOPSEC_REDIS_URL']) {
        $envVars['QOPSEC_REDIS_URL'] = $envVars['QOPSEC_REDIS_URL'] -replace 'REDIS_PASS', $envVars['QOPSEC_REDIS_PASSWORD']
    }
    if ($envVars['QOPSEC_RABBITMQ_URL']) {
        $envVars['QOPSEC_RABBITMQ_URL'] = $envVars['QOPSEC_RABBITMQ_URL'] -replace 'RABBIT_USER', $envVars['QOPSEC_RABBITMQ_USER']
        $envVars['QOPSEC_RABBITMQ_URL'] = $envVars['QOPSEC_RABBITMQ_URL'] -replace 'RABBIT_PASS', $envVars['QOPSEC_RABBITMQ_PASSWORD']
    }
    foreach ($k in $envVars.Keys) { Set-Item -Path "env:$k" -Value ([string]$envVars[$k]) }
    Write-Host '[OK] Segredos carregados' -ForegroundColor Green
} else {
    Write-Error "Segredos nao encontrados em $secretsFile. Rode o install.ps1 primeiro."; exit 1
}

# 2. ja esta rodando?
if ((Test-Path $pidFile) -and (Get-Process -Id (Get-Content $pidFile) -ErrorAction SilentlyContinue)) {
    Write-Host '[OK] Orquestrador ja esta rodando' -ForegroundColor Green
} else {
    New-Item -ItemType Directory -Path (Join-Path $InstallDir 'logs') -Force | Out-Null
    Write-Host '[..] Iniciando orquestrador (8090)...'
    $proc = Start-Process -FilePath $venvPython `
        -ArgumentList 'orchestrator.py' `
        -WorkingDirectory $InstallDir `
        -RedirectStandardOutput (Join-Path $InstallDir 'logs\qopsec.out.log') `
        -RedirectStandardError  (Join-Path $InstallDir 'logs\qopsec.err.log') `
        -PassThru -NoNewWindow
    Set-Content -Path $pidFile -Value $proc.Id
    Write-Host "[OK] Orquestrador PID $($proc.Id)" -ForegroundColor Green

    $ready = $false
    for ($i = 1; $i -le 24; $i++) {
        try { if ((Invoke-WebRequest -Uri 'http://127.0.0.1:8090/architecture/status' -UseBasicParsing -TimeoutSec 5).StatusCode -lt 400) { $ready = $true; break } } catch { }
        Start-Sleep -Seconds 5
    }
    if (-not $ready) { Write-Warning 'Orquestrador nao respondeu a tempo; veja logs\qopsec.err.log' }
    else { Write-Host '[OK] Orquestrador pronto' -ForegroundColor Green }
}

# 3. servicos
if (-not $SkipServices) {
    Write-Host '[..] Disparando /start/all...'
    try { Invoke-RestMethod -Uri 'http://127.0.0.1:8090/start/all' -Method Post -TimeoutSec 90 | Out-Null; Write-Host '[OK] Servicos iniciados' -ForegroundColor Green }
    catch { Write-Warning "Falha no /start/all: $($_.Exception.Message)" }
}

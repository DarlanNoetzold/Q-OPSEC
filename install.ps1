#!/usr/bin/env pwsh
#Requires -Version 5.1
<#
.SYNOPSIS
    Q-OPSEC Windows Installer (pacote)
.DESCRIPTION
    Instala tudo que o Q-OPSEC precisa para rodar no Windows:
      - Git, Python 3.13, JDK 21 (Temurin), Maven e Docker Desktop (via winget)
      - Ambiente virtual Python com dependencias de todos os submodulos
        (numpy/pandas/scikit-learn/torch/qiskit/kms-service/e oqs)
      - Infraestrutura Docker (Redis, RabbitMQ, Prometheus, Grafana,
        PostgreSQL, MongoDB e Mosquitto)
      - Orquestrador iniciado na porta 8090 com todos os servicos
    Uso (no CMD do Windows, com curl):
      curl -fsSL https://raw.githubusercontent.com/DarlanNoetzold/Q-OPSEC/package/windows/install.ps1 -o %TEMP%\qopsec-install.ps1 ^
        && powershell -NoProfile -ExecutionPolicy Bypass -File %TEMP%\qopsec-install.ps1
#>
param(
    [string]$InstallDir = (Join-Path $env:USERPROFILE 'Q-OPSEC'),
    [string]$Branch = 'package/windows',
    [string]$RepoUrl = 'https://github.com/DarlanNoetzold/Q-OPSEC.git',
    [string]$SelfUrl = 'https://raw.githubusercontent.com/DarlanNoetzold/Q-OPSEC/package/windows/install.ps1',
    [switch]$SkipDocker,
    [switch]$NoDockerDesktop
)

$ErrorActionPreference = 'Stop'
$ProgressPreference   = 'SilentlyContinue'

# ---------------------------------------------------------------- helpers
function Write-Step   { param([string]$Msg) Write-Host "`n=== $Msg ===" -ForegroundColor Cyan }
function Write-Ok     { param([string]$Msg) Write-Host "  [OK] $Msg" -ForegroundColor Green }
function Write-Warn   { param([string]$Msg) Write-Host "  [AVISO] $Msg" -ForegroundColor Yellow }
function Write-Fail   { param([string]$Msg) Write-Host "  [ERRO] $Msg" -ForegroundColor Red }

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p  = New-Object Security.Principal.WindowsPrincipal($id)
    return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function New-RandomPassword {
    param([int]$Length = 24)
    $bytes = New-Object byte[] ($Length)
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    return [Convert]::ToBase64String($bytes) -replace '[^a-zA-Z0-9]', 'A'
}

function Invoke-Checked {
    param([string]$Msg, [scriptblock]$Block)
    try { & $Block; Write-Ok $Msg }
    catch { throw "Falha ao executar '$Msg': $($_.Exception.Message)" }
}

function Wait-Health {
    param([string]$Url, [string]$Name, [int]$Attempts = 40)
    for ($i = 1; $i -le $Attempts; $i++) {
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400) {
                Write-Ok "$Name pronto (tentativa $i)"
                return $true
            }
        } catch { }
        Start-Sleep -Seconds 5
    }
    Write-Warn "$Name nao respondeu depois de $($Attempts * 5)s"
    return $false
}

# ---------------------------------------------------------------- elevacao
if (-not (Test-Admin)) {
    if ($PSCommandPath) {
        Write-Warn 'Sem privilegios de administrador. Pedindo elevacao...'
        try {
            Invoke-WebRequest -Uri $SelfUrl -OutFile "$env:TEMP\qopsec-install.ps1" -UseBasicParsing
            $argsLine = "-NoProfile -ExecutionPolicy Bypass -File `"$env:TEMP\qopsec-install.ps1`" -InstallDir `"$InstallDir`""
            if ($SkipDocker)     { $argsLine += ' -SkipDocker' }
            if ($NoDockerDesktop){ $argsLine += ' -NoDockerDesktop' }
            Start-Process powershell.exe -Verb RunAs -ArgumentList $argsLine
            Write-Host 'Reabra a janela elevada e acompanhe a instalacao la.' -ForegroundColor Yellow
            exit 0
        } catch {
            Write-Fail "Nao foi possivel elevar: $($_.Exception.Message)"
            Write-Fail 'Abra o PowerShell como Administrador e execute o script novamente.'
            exit 1
        }
    } else {
        Write-Fail 'Sem elevacao. Use o comando curl do README (que baixa o script e executa elevado),'
        Write-Fail 'ou abra o PowerShell como Administrador e rode o script novamente.'
        exit 1
    }
}

Write-Host '==================================================' -ForegroundColor Cyan
Write-Host '  Q-OPSEC - Instalador do pacote para Windows' -ForegroundColor Cyan
Write-Host '==================================================' -ForegroundColor Cyan

# ---------------------------------------------------------------- 1. dependencias de sistema (winget)
Write-Step '[1/8] Instalando dependencias de sistema (winget)'

$winget = Get-Command winget.exe -ErrorAction SilentlyContinue
if (-not $winget) {
    Write-Fail 'winget nao encontrado. Atualize o App Installer da Microsoft Store (Windows 10/11).'
    exit 1
}

$packages = @(
    'Git.Git',
    'Python.Python.3.13',
    'EclipseAdoptium.Temurin.21.JDK',
    'Apache.Maven'
)
if (-not $NoDockerDesktop) { $packages += 'Docker.DockerDesktop' }

$exeMap = @{
    'Git.Git'                      = 'git'
    'Python.Python.3.13'           = 'py'
    'EclipseAdoptium.Temurin.21.JDK' = 'java'
    'Apache.Maven'                 = 'mvn'
    'Docker.DockerDesktop'         = 'docker'
}

foreach ($pkg in $packages) {
    $exeName = $exeMap[$pkg]
    $existed = if ($exeName) { Get-Command $exeName -ErrorAction SilentlyContinue } else { $null }
    if ($existed) { Write-Ok "$pkg ja instalado"; continue }
    Write-Host "Instalando $pkg (pode demorar)..."
    & winget install --id $pkg --exact --silent --accept-source-agreements --accept-package-agreements | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "$pkg pode nao ter instalado (codigo $LASTEXITCODE; Docker Desktop costuma exigir reinicio)"
    } else {
        Write-Ok "$pkg instalado"
    }
}

# Atualiza PATH da sessao com JDK/Maven/Git recém-instalados (se ainda nao estiverem)
$pathsParaAdicionar = @()
$procFiles = @($env:ProgramFiles, ${env:ProgramFiles(x86)})
foreach ($pf in $procFiles) {
    if (-not $pf) { continue }
    $jdk = Get-ChildItem -Path $pf -Directory -Filter 'jdk-21*' -ErrorAction SilentlyContinue | Select-Object -First 1
    $adoptium = Join-Path $pf 'Eclipse Adoptium'
    if (-not $jdk -and (Test-Path $adoptium)) {
        $jdk = Get-ChildItem -Path $adoptium -Directory -Filter 'jdk-21*' -ErrorAction SilentlyContinue | Select-Object -First 1
    }
    if ($jdk) { $pathsParaAdicionar += (Join-Path $jdk.FullName 'bin') }
    $mvnHome = Join-Path $pf 'Apache'
    if (Test-Path $mvnHome) {
        $mvn = Get-ChildItem -Path $mvnHome -Directory -Filter 'maven*' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($mvn) { $pathsParaAdicionar += (Join-Path $mvn.FullName 'bin') }
    }
}
foreach ($p in $pathsParaAdicionar) {
    if ($p -and ($env:PATH -notlike "*$p*")) { $env:PATH = "$p;$env:PATH" }
}

$javaCheck = & 'java' -version 2>&1 | Select-Object -First 1
$mvnCheck  = & 'mvn'  -version 2>&1 | Select-Object -First 1
Write-Ok "Java: $javaCheck"
Write-Ok "Maven: $mvnCheck"

# ---------------------------------------------------------------- 2. clone do repositorio
Write-Step '[2/8] Baixando o codigo do Q-OPSEC'

if (Test-Path (Join-Path $InstallDir 'services.yaml')) {
    Write-Ok "Repositorio ja existe em $InstallDir (reutilizando)"
} else {
    Write-Host "Clonando $RepoUrl (branch $Branch) ..."
    if (-not (Test-Path (Split-Path $InstallDir -Parent))) {
        New-Item -ItemType Directory -Path (Split-Path $InstallDir -Parent) -Force | Out-Null
    }
    & git clone --depth 1 --branch $Branch $RepoUrl $InstallDir
    if ($LASTEXITCODE -ne 0) {
        Write-Fail 'Falha ao clonar o repositorio. Confira se a branch/URL estao publicas.'
        exit 1
    }
    Write-Ok "Clonado em $InstallDir"
}
Set-Location $InstallDir

# ---------------------------------------------------------------- 3. venv + dependencias Python
Write-Step '[3/8] Ambiente virtual Python e dependencias (pode demorar MUITO)'

$venvPath = Join-Path $InstallDir 'qopsec_env'
if (-not (Test-Path $venvPath)) {
    $pyCmd = $null
    try { $pyCmd = Get-Command py.exe -ErrorAction Stop } catch { }
    if ($pyCmd) {
        & py.exe -3.13 -m venv $venvPath 2>$null
        if ($LASTEXITCODE -ne 0) { & py.exe -m venv $venvPath }
    } else {
        $python = Get-Command python.exe -ErrorAction Stop
        & $python.Source -m venv $venvPath
    }
    if ($LASTEXITCODE -ne 0) { throw 'Nao foi possivel criar o ambiente virtual (venv)' }
    Write-Ok 'venv criado'
}
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
if (-not (Test-Path $venvPython)) { throw "venv sem python em $venvPath" }

& $venvPython -m pip install --upgrade pip setuptools wheel | Out-Null

Write-Host 'Instalando numpy/pandas/scipy/matplotlib/scikit-learn...'
& $venvPython -m pip install numpy==2.5.1 pandas==3.0.5 scipy==1.18.0 | Out-Null
& $venvPython -m pip install matplotlib seaborn scikit-learn==1.9.0 | Out-Null

Write-Host 'Instalando xgboost/catboost/lightgbm (compilacao pode demorar)...'
& $venvPython -m pip install xgboost catboost lightgbm | Out-Null

Write-Host 'Instalando torch (build CPU - mais rapido que o com CUDA)...'
& $venvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu | Out-Null

Write-Host 'Instalando qiskit e qiskit-aer...'
& $venvPython -m pip install qiskit==2.5.1 qiskit-aer==0.17.2 | Out-Null

Write-Host 'Instalando oqs (Post-Quantum)...'
& $venvPython -m pip install oqs 2>$null
if ($LASTEXITCODE -ne 0) { Write-Warn 'oqs nao instalado; o KMS pode degradar (tente instalar manualmente: pip install oqs)' }

Write-Host 'Instalando requirements raiz e de submodulos...'
$reqFiles = @(Get-ChildItem -Path $InstallDir -Recurse -Filter 'requirements*.txt' -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notlike '*qopsec_env*' -and $_.FullName -notlike '*liboqs*' })
foreach ($req in $reqFiles) {
    Write-Host "  -> $($req.FullName.Replace($InstallDir, '.'))"
    & $venvPython -m pip install -r $req.FullName | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Warn "pip falhou em $($req.Name)" }
}
Write-Ok 'Dependencias Python instaladas'

# ---------------------------------------------------------------- 4. services.yaml no host local
Write-Step '[4/8] Ajustando services.yaml para o host local'

$svcYaml = Join-Path $InstallDir 'services.yaml'
if (Test-Path "$svcYaml.orig") { Write-Ok 'backup ja existente (services.yaml.orig)' }
else {
    Copy-Item $svcYaml "$svcYaml.orig"
    (Get-Content $svcYaml -Raw) -replace '192\.168\.18\.18', '127.0.0.1' | Set-Content $svcYaml -NoNewline
    Write-Ok '192.168.18.18 substituido por 127.0.0.1 (backup em services.yaml.orig)'
}

# ---------------------------------------------------------------- 5. secrets locais
Write-Step '[5/8] Gerando segredos locais (Redis/RabbitMQ)'

$secretsDir  = Join-Path $env:USERPROFILE '.config\qopsec'
$secretsFile = Join-Path $secretsDir 'middleware.env'
New-Item -ItemType Directory -Path $secretsDir -Force | Out-Null

if (Test-Path $secretsFile) {
    Write-Ok "middleware.env ja existe em $secretsFile (mantido)"
} else {
    $redisPass  = New-RandomPassword
    $rabbitUser = 'qopsec'
    $rabbitPass = New-RandomPassword
    @(
        "QOPSEC_REDIS_PASSWORD=$redisPass",
        "QOPSEC_RABBITMQ_USER=$rabbitUser",
        "QOPSEC_RABBITMQ_PASSWORD=$rabbitPass",
        'QOPSEC_REDIS_URL=redis://:REDIS_PASS@127.0.0.1:6380/0',
        'QOPSEC_RABBITMQ_URL=amqp://RABBIT_USER:RABBIT_PASS@127.0.0.1:5673/%2F'
    ) | Set-Content $secretsFile -Encoding ascii
    Write-Ok "Segredos gerados em $secretsFile (NUNCA versionar esse arquivo)"
}

# ---------------------------------------------------------------- 6. infra Docker
Write-Step '[6/8] Infraestrutura Docker (Redis, RabbitMQ, Prometheus, Grafana, PostgreSQL, MongoDB, Mosquitto)'

if ($SkipDocker) {
    Write-Warn '-SkipDocker informado; pula containers. O orquestrador nao vai subir sozinho.'
} else {
    $docker = Get-Command docker.exe -ErrorAction SilentlyContinue
    if (-not $docker) {
        Write-Fail 'Docker nao encontrado. Instale o Docker Desktop (winget install --id Docker.DockerDesktop) e reinicie a sessao.'
        exit 1
    }

    Write-Host 'Verificando se o Docker Desktop esta rodando...'
    $dockerOk = $false
    for ($i = 1; $i -le 30; $i++) {
        docker info 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $dockerOk = $true; break }
        Write-Host "  aguardando daemon Docker ($i/30)..."
        Start-Sleep -Seconds 5
    }
    if (-not $dockerOk) {
        Write-Fail 'Daemon Docker nao responde. Abra o Docker Desktop, aguarde e rode de novo (ou use -SkipDocker).'
        exit 1
    }
    Write-Ok 'Docker Desktop operacional'

    $containers = @(
        @{ Name = 'qopsec-postgres';  Image = 'postgres:16';                Ports = '-p 5432:5432';   Run = @('-e', 'POSTGRES_PASSWORD=postgres') },
        @{ Name = 'qopsec-mongo';     Image = 'mongo:7';                    Ports = '-p 27017:27017'; Run = @('-e', 'MONGO_INITDB_ROOT_USERNAME=daily', '-e', 'MONGO_INITDB_ROOT_PASSWORD=daily123') },
        @{ Name = 'qopsec-mosquitto'; Image = 'eclipse-mosquitto:2';        Ports = '-p 1883:1883';   Run = @() }
    )
    foreach ($c in $containers) {
        $exists = docker ps -a --format '{{.Names}}' | Select-String -Quiet "^$($c.Name)$"
        if ($exists) {
            Write-Ok "container $($c.Name) ja existe"
        } else {
            Write-Host "Subindo $($c.Name) ($($c.Image))..."
            & docker run -d --name $c.Name --restart unless-stopped $c.Ports $c.Run $c.Image 2>$null | Out-Null
            if ($LASTEXITCODE -ne 0) { Write-Warn "falha ao subir $($c.Name)" }
            else { Write-Ok "$($c.Name) no ar" }
        }
    }

    Write-Host 'Criando bancos PostgreSQL dos servicos Java...'
    foreach ($db in @('context_db', 'intercept_db', 'validation_send')) {
        $exists = docker exec qopsec-postgres psql -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$db'" 2>$null | Select-String -Quiet '^1$'
        if (-not $exists) {
            docker exec qopsec-postgres psql -U postgres -c "CREATE DATABASE $db" 2>$null | Out-Null
            Write-Ok "banco $db criado"
        }
    }
}

# ---------------------------------------------------------------- 7. iniciar orquestrador
Write-Step '[7/8] Iniciando orquestrador (porta 8090)'

$startupScript = Join-Path $InstallDir 'start_qopsec.ps1'
if (Test-Path $startupScript) { & $startupScript -SkipServices }
else { Write-Warn 'start_qopsec.ps1 nao encontrado; inicie o orquestrador manualmente' }

$orchReady = Wait-Health -Url 'http://127.0.0.1:8090/architecture/status' -Name 'Orquestrador' -Attempts 24
if (-not $orchReady) { Write-Fail 'Orquestrador nao subiu; veja logs/qopsec.out.log'; exit 1 }

Write-Host 'Disparando /start/all...'
try {
    Invoke-RestMethod -Uri 'http://127.0.0.1:8090/start/all' -Method Post -TimeoutSec 60 | Out-Null
    Write-Ok '/start/all disparado'
} catch {
    Write-Warn "Falha no /start/all: $($_.Exception.Message)"
}

# ---------------------------------------------------------------- 8. validacao final
Write-Step '[8/8] Validando os servicos'

$healths = @(
    @{ Url = 'http://127.0.0.1:65534/actuator/health'; Nome = 'context_api (Spring, 65534)' },
    @{ Url = 'http://127.0.0.1:8080/actuator/health';  Nome = 'interceptor_api (Spring, 8080)' },
    @{ Url = 'http://127.0.0.1:8005/actuator/health';  Nome = 'validation_send_api (Spring, 8005)' },
    @{ Url = 'http://127.0.0.1:8001/docs';             Nome = 'handshake_negotiator (8001)' },
    @{ Url = 'http://127.0.0.1:8003/health';           Nome = 'key_destination_engine (8003)' },
    @{ Url = 'http://127.0.0.1:8004/health';           Nome = 'crypto_module (8004)' },
    @{ Url = 'http://127.0.0.1:8000/risk/health';      Nome = 'risk_service (8000)' },
    @{ Url = 'http://127.0.0.1:8083/confidentiality/health'; Nome = 'confiability_service (8083)' },
    @{ Url = 'http://127.0.0.1:8088/api/v1/health';    Nome = 'classification_agent (8088)' },
    @{ Url = 'http://127.0.0.1:9009/docs';             Nome = 'rl_engine (9009)' }
)
foreach ($h in $healths) {
    $resp = Invoke-WebRequest -Uri $h.Url -UseBasicParsing -TimeoutSec 10 -ErrorAction SilentlyContinue
    if ($resp -and $resp.StatusCode -lt 400) { Write-Ok $h.Nome } else { Write-Warn "$($h.Nome) fora do ar" }
}

Write-Host ''
Write-Host '==================================================' -ForegroundColor Green
Write-Host '  INSTALACAO Q-OPSEC (WINDOWS) CONCLUIDA!' -ForegroundColor Green
Write-Host '==================================================' -ForegroundColor Green
Write-Host "  Orquestrador:  http://127.0.0.1:8090"
Write-Host "  Painel:        http://127.0.0.1:3001  (Grafana, login admin)"
Write-Host "  RabbitMQ mgmt: http://127.0.0.1:15673 (user: qopsec)"
Write-Host "  Chaves/creds:  $secretsFile"
Write-Host "  Para reiniciar: powershell -File $InstallDir\start_qopsec.ps1"
Write-Host "  Para parar:     powershell -File $InstallDir\stop_qopsec.ps1"
Write-Host ''
Write-Host 'OBS: o RabbitMQ/Redis do orquestrador sobem via containers gerenciados'
Write-Host "por ele mesmo (labels com.qopsec.managed-by). Se o Docker Desktop foi"
Write-Host 'instalado agora, talvez seja preciso reiniciar o Windows uma vez.'

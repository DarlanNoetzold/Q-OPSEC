# Tutorial de instalação — Q-OPSEC no Windows (`package/windows`)

Guia passo a passo para instalar, verificar, operar e atualizar o Q-OPSEC no
Windows 10/11 usando a branch de distribuição `package/windows`.

Tempo estimado: **15–40 min** (a maior parte é download e compilação de
dependências Python). Espaço em disco: **~15 GB** livres.

---

## 1. Requisitos

| Item | Mínimo |
| --- | --- |
| Sistema | Windows 10 (21H2+) ou Windows 11, x64 |
| Privilégios | conta de Administrador (o instalador pede elevação via UAC) |
| Disco | 15 GB livres |
| Memória | 8 GB (16 GB recomendado) |
| Virtualização | habilitada na BIOS/UEFI (necessária para o Docker Desktop/WSL2) |
| Programas | `curl` (nativo no Windows 10+) e **App Installer** (winget) da Microsoft Store |
| Rede | saída para `github.com`, `pypi.org`, `winget` e Docker Hub |

Portas que precisam estar livres: **8090, 8000–8005, 8080, 8083, 8088, 9009,
65534, 5432, 6380, 5673, 15673, 9091, 3001, 1883, 27017**.

---

## 2. Instalação em um comando

### 2.1 CMD como Administrador (recomendado)

```cmd
curl -fsSL https://raw.githubusercontent.com/DarlanNoetzold/Q-OPSEC/package/windows/install.ps1 -o %TEMP%\qopsec-install.ps1 && powershell -NoProfile -ExecutionPolicy Bypass -File %TEMP%\qopsec-install.ps1
```

### 2.2 PowerShell 7

```powershell
irm https://raw.githubusercontent.com/DarlanNoetzold/Q-OPSEC/package/windows/install.ps1 | iex
```

### 2.3 PowerShell 5.1 (pipe)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/DarlanNoetzold/Q-OPSEC/package/windows/install.ps1 | iex"
```

Se a janela não estiver elevada, o instalador baixa a si mesmo para
`%TEMP%\qopsec-install.ps1` e se reabre com `Start-Process -Verb RunAs`. A
instalação continua na janela elevada — acompanhe por ela.

---

## 3. O que o instalador faz

Oito etapas, cada uma com `[OK]`/`[AVISO]`:

| Etapa | Ação |
| --- | --- |
| 1/8 | `winget install` de `Git.Git`, `Python.Python.3.13`, `EclipseAdoptium.Temurin.21.JDK`, `Apache.Maven` e `Docker.DockerDesktop` (pula o que já existe) e ajusta o `PATH` da sessão para JDK/Maven |
| 2/8 | Clona a branch `package/windows` em `%USERPROFILE%\Q-OPSEC` (reutiliza se já existir) |
| 3/8 | Cria o venv `qopsec_env` e instala numpy/pandas/scipy/matplotlib/scikit-learn, xgboost/catboost/lightgbm, torch (CPU), qiskit/qiskit-aer, `oqs` e todos os `requirements*.txt` |
| 4/8 | Copia `services.yaml` para `services.yaml.orig` e troca `192.168.18.18` por `127.0.0.1` |
| 5/8 | Gera `%USERPROFILE%\.config\qopsec\middleware.env` com senhas aleatórias de Redis e RabbitMQ |
| 6/8 | Sobe `qopsec-postgres` (16), `qopsec-mongo` (7) e `qopsec-mosquitto` (2) e cria `context_db`, `intercept_db` e `validation_send` |
| 7/8 | Inicia o orquestrador na 8090 (`start_qopsec.ps1 -SkipServices`) e dispara `POST /start/all` |
| 8/8 | Testa o `/health` dos dez serviços e o `/actuator/health` das três APIs Java |

Redis, RabbitMQ, Prometheus e Grafana sobem como containers gerenciados pelo
orquestrador (label `com.qopsec.managed-by=orchestrator`), presos a `127.0.0.1`.

> Se o Docker Desktop for instalado nesta execução, **reinicie o Windows** e rode
> `start_qopsec.ps1` depois. O WSL2 também pode pedir um reboot.

---

## 4. Instalação com parâmetros

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$env:TEMP\qopsec-install.ps1" `
  -InstallDir "D:\Q-OPSEC" `
  -Branch "package/windows" `
  -SkipDocker
```

| Parâmetro | Padrão | Para quê |
| --- | --- | --- |
| `-InstallDir` | `%USERPROFILE%\Q-OPSEC` | diretório de instalação |
| `-Branch` | `package/windows` | branch clonada |
| `-RepoUrl` | `https://github.com/DarlanNoetzold/Q-OPSEC.git` | fork espelho |
| `-SkipDocker` | desligado | pula os containers de infra |
| `-NoDockerDesktop` | desligado | não instala o Docker Desktop via winget |

---

## 5. Verificação pós-instalação

```powershell
Invoke-RestMethod http://127.0.0.1:8090/architecture/status | ConvertTo-Json -Depth 5
```

Esperado: quatro blocos (`pipeline`, `resilience`, `rate_limit`, `tracing`), com
`rate_limit.backend` = `redis` e `tracing.enabled` = `true`.

Serviços e portas:

| Serviço | URL |
| --- | --- |
| Orquestrador | http://127.0.0.1:8090 |
| Grafana | http://127.0.0.1:3001 |
| Prometheus | http://127.0.0.1:9091 |
| RabbitMQ mgmt | http://127.0.0.1:15673 (usuário `qopsec`, senha em `middleware.env`) |
| risk_service | http://127.0.0.1:8000/health |
| handshake_negotiator | http://127.0.0.1:8001/health |
| kms_service | http://127.0.0.1:8002/health |
| key_destination_engine | http://127.0.0.1:8003/health |
| crypto_module | http://127.0.0.1:8004/health |
| validation_send_api | http://127.0.0.1:8005/actuator/health |
| interceptor_api | http://127.0.0.1:8080/actuator/health |
| confiability_service | http://127.0.0.1:8083/health |
| classification_agent | http://127.0.0.1:8088/api/v1/health |
| rl_engine | http://127.0.0.1:9009/health |
| context_api | http://127.0.0.1:65534/actuator/health |

Pipeline completo de ponta a ponta:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8090/run-pipeline -Method Post `
  -ContentType 'application/json' `
  -Body '{"source":"10.0.0.1","destination":"10.0.0.2"}' | ConvertTo-Json -Depth 5
```

Esperado: `status = succeeded`, 11 steps e `failed_steps` vazio.

---

## 6. Operação do dia a dia

```powershell
powershell -File $env:USERPROFILE\Q-OPSEC\start_qopsec.ps1                    # sobe tudo
powershell -File $env:USERPROFILE\Q-OPSEC\start_qopsec.ps1 -SkipServices      # só o orquestrador
powershell -File $env:USERPROFILE\Q-OPSEC\stop_qopsec.ps1                     # para serviços e orquestrador
powershell -File $env:USERPROFILE\Q-OPSEC\stop_qopsec.ps1 -StopContainers     # remove postgres/mongo/mosquitto
```

Logs:

| Arquivo | Conteúdo |
| --- | --- |
| `%USERPROFILE%\Q-OPSEC\logs\qopsec.out.log` | stdout do orquestrador |
| `%USERPROFILE%\Q-OPSEC\logs\qopsec.err.log` | stderr do orquestrador |
| `%USERPROFILE%\Q-OPSEC\logs\<serviço>.log` | log de cada serviço de processo |
| `%USERPROFILE%\Q-OPSEC\logs\orchestrator.pid` | PID do orquestrador |

Serviço isolado:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8090/services/kms_service/start -Method Post
Invoke-RestMethod -Uri http://127.0.0.1:8090/services/kms_service/stop  -Method Post
Invoke-RestMethod -Uri http://127.0.0.1:8090/services | ConvertTo-Json -Depth 5
```

---

## 7. Atualização

O instalador reutiliza o repositório existente, mas **não** faz `pull`:

```powershell
cd $env:USERPROFILE\Q-OPSEC
git fetch origin package/windows
git checkout package/windows
git reset --hard origin/package/windows
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

O `reset --hard` descarta a troca de IP feita em `services.yaml`; o `install.ps1`
reaplica a substituição para `127.0.0.1` em toda execução. O backup original
continua em `services.yaml.orig`.

---

## 8. Problemas comuns

| Sintoma | Causa | Correção |
| --- | --- | --- |
| `winget nao encontrado` | App Installer ausente/desatualizado | atualize o **App Installer** na Microsoft Store |
| `... não pode ser carregado porque a execução de scripts foi desabilitada` | ExecutionPolicy | use `-ExecutionPolicy Bypass` como nos comandos acima |
| `Daemon Docker nao responde` | Docker Desktop fechado ou WSL2 sem reboot | abra o Docker Desktop, aguarde o ícone estabilizar e rode de novo |
| `WSL 2 installation is incomplete` | recurso WSL não habilitado | `wsl --install` em PowerShell admin e reinicie |
| `Orquestrador nao subiu` | porta 8090 ocupada | `Get-NetTCPConnection -LocalPort 8090` e `Get-Content logs\qopsec.err.log -Tail 50` |
| `oqs nao instalado` | wheel indisponível para a versão do Python | `qopsec_env\Scripts\pip.exe install oqs`; se falhar, o KMS degrada para software |
| APIs Java fora do ar | Postgres não subiu ou banco ausente | `docker logs qopsec-postgres` e confira os três bancos |
| Firewall pedindo liberação | Docker Desktop/publicação de portas | permita apenas rede privada; os serviços escutam em `127.0.0.1` |

---

## 9. Desinstalação

```powershell
powershell -File $env:USERPROFILE\Q-OPSEC\stop_qopsec.ps1 -StopContainers
docker rm -f qopsec-redis qopsec-rabbitmq qopsec-prom qopsec-graf 2>$null
Remove-Item -Recurse -Force $env:USERPROFILE\Q-OPSEC
Remove-Item -Recurse -Force $env:USERPROFILE\.config\qopsec
```

Para listar containers gerenciados que sobraram:

```powershell
docker ps -a --filter label=com.qopsec.managed-by=orchestrator
```

---

## 10. Segurança

- `%USERPROFILE%\.config\qopsec\middleware.env` guarda as senhas de Redis e
  RabbitMQ. **Nunca** versione esse arquivo.
- Redis e RabbitMQ escutam apenas em `127.0.0.1`; o orquestrador recusa subir
  esses containers com bind em `0.0.0.0`.
- O Postgres do instalador usa `POSTGRES_PASSWORD=postgres` e o Mongo usa
  `daily/daily123`: credenciais de desenvolvimento. Troque antes de expor em rede.

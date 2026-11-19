# Tutorial de instalação — Q-OPSEC no Linux (`package/linux`)

Guia passo a passo para instalar, verificar, operar e atualizar o Q-OPSEC em
Debian/Ubuntu usando a branch de distribuição `package/linux`.

Tempo estimado: **20–40 min** (a maior parte é download e compilação de
dependências Python). Espaço em disco: **~10 GB** livres.

---

## 1. Requisitos

| Item | Mínimo |
| --- | --- |
| Distribuição | Debian 12 / Ubuntu 22.04 ou 24.04 (qualquer derivada com `apt`) |
| Arquitetura | x86_64 ou arm64 |
| Memória | 8 GB (16 GB recomendado para treinar modelos) |
| Disco | 10 GB livres |
| Acesso | usuário com `sudo` (ou root) |
| Rede | saída para `github.com`, `pypi.org` e Docker Hub |
| Programas | `curl` e `git` (o instalador instala o resto) |

Portas que precisam estar livres: **8090, 8000–8005, 8080, 8083, 8088, 9009,
65534, 5432, 6380, 5673, 15673, 9091, 3001, 1883, 27017**.

---

## 2. Instalação em um comando

### 2.1 Forma direta (pipe)

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/DarlanNoetzold/Q-OPSEC/package/linux/install.sh)
```

### 2.2 Forma em duas etapas (recomendada: permite ler o script antes)

```bash
curl -fsSL https://raw.githubusercontent.com/DarlanNoetzold/Q-OPSEC/package/linux/install.sh -o /tmp/qopsec-install.sh
less /tmp/qopsec-install.sh          # inspeção opcional
bash /tmp/qopsec-install.sh
```

O instalador usa `sudo` quando necessário. Se você já é root, ele não pede senha.

---

## 3. O que o instalador faz

O script executa nove etapas e imprime `[OK]`/`[AVISO]` em cada uma:

| Etapa | Ação |
| --- | --- |
| 1/9 | `apt-get install` de `build-essential`, `cmake`, `ninja-build`, `libssl-dev`, `python3-dev/venv/pip`, `openjdk-21-jdk`, `maven`, `docker.io`, `docker-compose-v2`, `mosquitto` e adiciona o usuário ao grupo `docker` |
| 2/9 | Clona a branch `package/linux` em `~/Q-OPSEC` (reutiliza o diretório se já existir) |
| 3/9 | Copia `services.yaml` para `services.yaml.orig` e troca `192.168.18.18` por `127.0.0.1` |
| 4/9 | Gera `~/.config/qopsec/middleware.env` (0600) com senhas aleatórias de Redis e RabbitMQ |
| 5/9 | Cria o venv `qopsec_env` e instala numpy/pandas/scipy/scikit-learn, xgboost/catboost/lightgbm, torch (CPU), qiskit/qiskit-aer, `oqs` e todos os `requirements*.txt` do repositório |
| 6/9 | Sobe `qopsec-postgres` (16), `qopsec-mongo` (7) e `qopsec-mosquitto` (2) e cria os bancos `context_db`, `intercept_db` e `validation_send` |
| 7/9 | Inicia o orquestrador na 8090 (`start_qopsec_linux.sh --skip-services`) e dispara `POST /start/all` |
| 8/9 | Testa o `/health` dos dez serviços e o `/actuator/health` das três APIs Java |
| 9/9 | Imprime URLs, caminho dos segredos e comandos de start/stop |

Redis, RabbitMQ, Prometheus e Grafana **não** são criados pelo instalador: eles
sobem como containers gerenciados pelo próprio orquestrador (label
`com.qopsec.managed-by=orchestrator`), com portas presas a `127.0.0.1`.

---

## 4. Instalação com parâmetros

Todas as variáveis são opcionais e vão antes do `bash`:

```bash
INSTALL_DIR=/opt/qopsec \
BRANCH=package/linux \
REPO_URL=https://github.com/DarlanNoetzold/Q-OPSEC.git \
SKIP_DOCKER=0 \
bash /tmp/qopsec-install.sh
```

| Variável | Padrão | Para quê |
| --- | --- | --- |
| `INSTALL_DIR` | `$HOME/Q-OPSEC` | diretório de instalação |
| `BRANCH` | `package/linux` | branch clonada |
| `REPO_URL` | `https://github.com/DarlanNoetzold/Q-OPSEC.git` | fork espelho |
| `SKIP_DOCKER` | `0` | `1` pula os containers de infra (Postgres/Mongo/Mosquitto) |

---

## 5. Verificação pós-instalação

```bash
curl -s http://127.0.0.1:8090/architecture/status | python3 -m json.tool
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

```bash
curl -s -X POST http://127.0.0.1:8090/run-pipeline \
  -H 'Content-Type: application/json' \
  -d '{"source":"10.0.0.1","destination":"10.0.0.2"}' | python3 -m json.tool
```

Esperado: `"status": "succeeded"`, `"steps"` com 11 entradas e `"failed_steps": []`.

---

## 6. Operação do dia a dia

```bash
bash ~/Q-OPSEC/start_qopsec_linux.sh                      # sobe orquestrador + serviços
bash ~/Q-OPSEC/start_qopsec_linux.sh --skip-services      # só o orquestrador
bash ~/Q-OPSEC/stop_qopsec_linux.sh                       # para serviços e orquestrador
bash ~/Q-OPSEC/stop_qopsec_linux.sh --remove-containers   # remove também postgres/mongo/mosquitto
```

Logs:

| Arquivo | Conteúdo |
| --- | --- |
| `~/Q-OPSEC/logs/qopsec.out.log` | stdout do orquestrador |
| `~/Q-OPSEC/logs/qopsec.err.log` | stderr do orquestrador |
| `~/Q-OPSEC/logs/<serviço>.log` | log de cada serviço de processo |
| `~/Q-OPSEC/logs/orchestrator.pid` | PID do orquestrador |

Subir ou parar um serviço isolado:

```bash
curl -X POST http://127.0.0.1:8090/services/kms_service/start
curl -X POST http://127.0.0.1:8090/services/kms_service/stop
curl -s  http://127.0.0.1:8090/services | python3 -m json.tool
```

---

## 7. Atualização

O instalador reutiliza o repositório existente, mas **não** faz `pull`. Para
atualizar para o último commit da branch:

```bash
cd ~/Q-OPSEC
git fetch origin package/linux
git checkout package/linux
git reset --hard origin/package/linux
bash install.sh
```

O `reset --hard` descarta a troca de IP feita em `services.yaml`; o `install.sh`
reaplica a substituição para `127.0.0.1` em toda execução. O backup original
continua em `services.yaml.orig`.

---

## 8. Problemas comuns

| Sintoma | Causa | Correção |
| --- | --- | --- |
| `permission denied` ao usar `docker` | usuário ainda não está no grupo `docker` | `newgrp docker` ou reabra a sessão |
| `daemon Docker nao responde` | serviço parado | `sudo systemctl start docker` e rode o instalador de novo |
| `Orquestrador nao subiu` | porta 8090 ocupada ou erro no venv | `ss -ltnp \| grep 8090` e `tail -50 ~/Q-OPSEC/logs/qopsec.err.log` |
| `oqs nao instalou` | falta `libssl-dev`/compilador | `sudo apt-get install -y build-essential libssl-dev cmake ninja-build` e `pip install oqs` no venv |
| APIs Java fora do ar | Postgres não subiu ou banco ausente | `docker logs qopsec-postgres` e confira `context_db`, `intercept_db`, `validation_send` |
| `pip` falha em `torch` | índice CPU inacessível | `~/Q-OPSEC/qopsec_env/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Serviço reinicia em loop | dependência ainda subindo | aguarde 60 s e confira `curl :8090/services` |

---

## 9. Desinstalação

```bash
bash ~/Q-OPSEC/stop_qopsec_linux.sh --remove-containers
docker rm -f qopsec-redis qopsec-rabbitmq qopsec-prom qopsec-graf 2>/dev/null
rm -rf ~/Q-OPSEC ~/.config/qopsec
```

Os containers gerenciados pelo orquestrador usam o label
`com.qopsec.managed-by=orchestrator`; para listar o que sobrou:

```bash
docker ps -a --filter label=com.qopsec.managed-by=orchestrator
```

---

## 10. Segurança

- `~/.config/qopsec/middleware.env` guarda as senhas de Redis e RabbitMQ com
  permissão `0600`. **Nunca** versione esse arquivo.
- Redis e RabbitMQ escutam apenas em `127.0.0.1`; o orquestrador recusa subir
  esses containers com bind em `0.0.0.0`.
- O Postgres do instalador usa `POSTGRES_PASSWORD=postgres` e o Mongo usa
  `daily/daily123`: são credenciais de desenvolvimento. Para exposição em rede,
  troque-as e mova para um secret store.

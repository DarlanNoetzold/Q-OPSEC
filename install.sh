#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
#  Q-OPSEC - Instalador do pacote para LINUX
# ---------------------------------------------------------------------------
#  Instala tudo que o Q-OPSEC precisa para rodar em qualquer distribuição
#  Debian/Ubuntu:
#    - dependências de sistema (build-essential cmake ninja git libssl python3
#      venv, JDK 21, Maven, Docker/Compose, Mosquitto)
#    - ambiente virtual Python com todas as dependências (numpy/pandas/sklearn/
#      torch/qiskit/oqs pós-quântica + todos requirements de sub-módulos)
#    - ajusta services.yaml para 127.0.0.1 e gera segredos locais
#    - sobe infraestrutura Docker (Redis, RabbitMQ, Prometheus, Grafana,
#      PostgreSQL, MongoDB, Mosquitto)
#    - inicia o orquestrador (porta 8090) e dispara /start/all
#
#  Uso (one-liner, bash):
#    bash <(curl -fsSL https://raw.githubusercontent.com/DarlanNoetzold/Q-OPSEC/package/linux/install.sh)
#  ou
#    curl -fsSL https://raw.githubusercontent.com/DarlanNoetzold/Q-OPSEC/package/linux/install.sh -o /tmp/qopsec-install.sh \
#      && bash /tmp/qopsec-install.sh
# ===========================================================================

# ------------------------------------------------------------- config
INSTALL_DIR="${INSTALL_DIR:-$HOME/Q-OPSEC}"
BRANCH="${BRANCH:-package/linux}"
REPO_URL="${REPO_URL:-https://github.com/DarlanNoetzold/Q-OPSEC.git}"
SKIP_DOCKER="${SKIP_DOCKER:-0}"

# ------------------------------------------------------------- utils
step()   { echo; echo "=== $1 ==="; }
ok()     { echo "  [OK] $1"; }
warn()   { echo "  [AVISO] $1" >&2; }
fail()   { echo "  [ERRO] $1" >&2; exit 1; }
sudorun() {
    if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi
}
randpass() { head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 24; }
wait_health() {  # wait_health URL NOME [TENTATIVAS]
    local url="$1" nome="$2" attempts="${3:-40}" i=1
    while [ "$i" -le "$attempts" ]; do
        if curl -sf -o /dev/null --max-time 5 "$url"; then ok "$nome pronto (tentativa $i)"; return 0; fi
        echo "  aguardando $nome ($i/$attempts)..."; sleep 5; i=$((i+1))
    done
    warn "$nome nao respondeu apos $((attempts*5))s"; return 1
}

# ------------------------------------------------------------- root?
if [ "$(id -u)" -ne 0 ]; then
    if [ "$(command -v sudo)" ]; then
        echo "Sem root; o instalador vai usar sudo quando precisar."
    else
        fail "Rode como root ou instale sudo."
    fi
fi

[ -z "$(command -v curl)" ] || [ -z "$(command -v git)" ] || true
command -v curl >/dev/null 2>&1 || { echo "curl obrigatorio"; exit 1; }
command -v git  >/dev/null 2>&1 || { echo "git obrigatorio"; exit 1; }

step "[1/9] Dependencias de sistema (apt)"

# Base + build nativo + Java/Maven/Docker/MQTT
sudorun apt-get update -qq
sudorun env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl git \
    build-essential cmake ninja-build \
    libssl-dev \
    python3-dev python3-venv python3-pip \
    openjdk-21-jdk maven \
    docker.io docker-compose-v2 \
    mosquitto mosquitto-clients \
    ca-certificates

# Docker Compose v2 (plugin) pode vir separado em alguns distros
if ! command -v docker >/dev/null 2>&1; then
    fail "Docker nao instalado. Instale docker-ce ou docker.io e rode de novo."
fi
if ! docker compose version >/dev/null 2>&1; then
    sudorun apt-get install -y docker-compose-v2 2>/dev/null || warn "docker compose v2 ausente (ignore se nao for usar)"
fi
ok "dependencias de sistema instaladas"

# Garante que o usuario possa usar docker (evita sudo em todo comando)
if [ "$(id -u)" -ne 0 ]; then
    sudorun usermod -aG docker "$USER" 2>/dev/null || true
fi

step "[2/9] Carregando codigo do Q-OPSEC ($BRANCH)"
if [ -f "$INSTALL_DIR/services.yaml" ]; then
    echo "  repositorio ja existe em $INSTALL_DIR (reutilizando)"
    cd "$INSTALL_DIR"
    git fetch origin "$BRANCH" >/dev/null 2>&1 || true
    git checkout "$BRANCH" 2>/dev/null || true
else
    mkdir -p "$(dirname "$INSTALL_DIR")"
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    ok "clonado em $INSTALL_DIR"
fi

step "[3/9] Ajustando services.yaml para host local"
SVY="$INSTALL_DIR/services.yaml"
if [ -f "$SVY.orig" ]; then
    echo "  backup ja existente (services.yaml.orig)"
else
    cp "$SVY" "$SVY.orig"
    sed -i 's/192\.168\.18\.18/127.0.0.1/g' "$SVY"
    ok "base_url 192.168.18.18 -> 127.0.0.1"
fi

step "[4/9] Segredos locais (Redis/RabbitMQ)"
SECRETS_DIR="$HOME/.config/qopsec"
SECRETS_FILE="$SECRETS_DIR/middleware.env"
mkdir -p "$SECRETS_DIR"
if [ -f "$SECRETS_FILE" ]; then
    ok "$SECRETS_FILE ja existe (mantido)"
else
    REDIS_PASS="$(randpass)"
    RABBIT_USER="qopsec"
    RABBIT_PASS="$(randpass)"
    cat > "$SECRETS_FILE" <<EOF
QOPSEC_REDIS_PASSWORD=$REDIS_PASS
QOPSEC_RABBITMQ_USER=$RABBIT_USER
QOPSEC_RABBITMQ_PASSWORD=$RABBIT_PASS
QOPSEC_REDIS_URL=redis://:$REDIS_PASS@127.0.0.1:6380/0
QOPSEC_RABBITMQ_URL=amqp://$RABBIT_USER:$RABBIT_PASS@127.0.0.1:5673/%2F
EOF
    chmod 600 "$SECRETS_FILE"
    ok "segredos gerados em $SECRETS_FILE (NUNCA versionar)"
fi

step "[5/9] Ambiente virtual Python e dependencias (pode demorar MUITO)"
VENV="$INSTALL_DIR/qopsec_env"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    ok "venv criado"
fi
VPY="$VENV/bin/python"
$VPY -m pip install --upgrade pip setuptools wheel -q

echo "  numpy/pandas/scipy/matplotlib/scikit-learn..."
$VPY -m pip install -q numpy==2.5.1 pandas==3.0.5 scipy==1.18.0 matplotlib seaborn scikit-learn==1.9.0

echo "  xgboost/catboost/lightgbm..."
$VPY -m pip install -q xgboost catboost lightgbm || warn "falha em xgboost/catboost/lightgbm"

echo "  torch (CPU)..."
$VPY -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu || warn "falha no torch (CUDA opcional)"

echo "  qiskit/qiskit-aer..."
$VPY -m pip install -q qiskit==2.5.1 qiskit-aer==0.17.2 || warn "falha no qiskit"

echo "  oqs (post-quantico)..."
$VPY -m pip install -q oqs 2>/dev/null || warn "oqs nao instalou; KMS pode degradar"

echo "  requirements raiz e submodulos..."
mapfile -t REQ_FILES < <(find "$INSTALL_DIR" -name 'requirements*.txt' -not -path '*qopsec_env*' -not -path '*liboqs*' 2>/dev/null)
for req in "${REQ_FILES[@]}"; do
    echo "    -> ${req#$INSTALL_DIR}"
    $VPY -m pip install -q -r "$req" || warn "pip falhou em $(basename "$req")"
done
ok "dependencias Python instaladas"

# --------------------------------------------------------------- 6. Infra Docker
step "[6/9] Infraestrutura Docker (postgres/mongo/mosquitto)"
if [ "$SKIP_DOCKER" = "1" ] || [ "$SKIP_DOCKER" = "true" ]; then
    warn "-SKIP_DOCKER ativo; pula containers de infra (orquestrador pode nao subir sozinho)"
else
    echo "  aguardando daemon Docker..."
    docker_ok=0
    for i in $(seq 1 30); do
        if docker info >/dev/null 2>&1; then docker_ok=1; break; fi
        echo "    aguardando daemon ($i/30)..."; sleep 5
    done
    [ "$docker_ok" = "1" ] || fail "daemon Docker nao responde. Inicie o serviço (systemctl start docker) e rode de novo."

    # --- postgres ---
    if docker ps -a --format '{{.Names}}' | grep -qx qopsec-postgres; then
        echo "  container qopsec-postgres ja existe"
    else
        echo "  subindo qopsec-postgres (postgres:16)..."
        docker run -d --name qopsec-postgres --restart unless-stopped \
            -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16 >/dev/null \
            || warn "falha ao subir qopsec-postgres"
    fi
    echo "  aguardando postgres..."
    for i in $(seq 1 30); do
        if docker exec qopsec-postgres pg_isready -U postgres >/dev/null 2>&1; then break; fi
        sleep 3
    done
    echo "  criando bancos dos servicos Java..."
    for db in context_db intercept_db validation_send; do
        if ! docker exec qopsec-postgres psql -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$db'" 2>/dev/null | grep -q 1; then
            docker exec qopsec-postgres psql -U postgres -c "CREATE DATABASE $db" >/dev/null 2>&1 \
                && echo "    banco $db criado" || warn "falha ao criar banco $db"
        fi
    done

    # --- mongo ---
    if docker ps -a --format '{{.Names}}' | grep -qx qopsec-mongo; then
        echo "  container qopsec-mongo ja existe"
    else
        echo "  subindo qopsec-mongo (mongo:7)..."
        docker run -d --name qopsec-mongo --restart unless-stopped \
            -p 27017:27017 \
            -e MONGO_INITDB_ROOT_USERNAME=daily \
            -e MONGO_INITDB_ROOT_PASSWORD=daily123 \
            mongo:7 >/dev/null || warn "falha ao subir qopsec-mongo"
    fi

    # --- mosquitto ---
    if docker ps -a --format '{{.Names}}' | grep -qx qopsec-mosquitto; then
        echo "  container qopsec-mosquitto ja existe"
    else
        echo "  subindo qopsec-mosquitto (eclipse-mosquitto:2)..."
        docker run -d --name qopsec-mosquitto --restart unless-stopped \
            -p 1883:1883 eclipse-mosquitto:2 >/dev/null || warn "falha ao subir qopsec-mosquitto"
    fi
    ok "infra docker pronta"
fi

# --------------------------------------------------------------- 7. Iniciar orquestrador
step "[7/9] Iniciando orquestrador (porta 8090)"
mkdir -p "$INSTALL_DIR/logs"
STUP="$INSTALL_DIR/start_qopsec_linux.sh"
if [ -f "$STUP" ]; then (cd "$INSTALL_DIR" && bash "$STUP" --skip-services); fi

if wait_health "http://127.0.0.1:8090/architecture/status" "Orquestrador" 24; then
    echo "  disparando /start/all..."
    curl -sf -X POST "http://127.0.0.1:8090/start/all" >/dev/null 2>&1 \
        && ok "/start/all disparado" || warn "falha no /start/all"
else
    fail "Orquestrador nao subiu; veja $INSTALL_DIR/logs/qopsec.out.log"
fi

# --------------------------------------------------------------- 8. Validar servicos Java
step "[8/9] Validando servicos"
healths=(
    "http://127.0.0.1:65534/actuator/health|context_api (Spring, 65534)"
    "http://127.0.0.1:8080/actuator/health|interceptor_api (Spring, 8080)"
    "http://127.0.0.1:8005/actuator/health|validation_send_api (Spring, 8005)"
    "http://127.0.0.1:8001/docs|handshake_negotiator (8001)"
    "http://127.0.0.1:8003/health|key_destination_engine (8003)"
    "http://127.0.0.1:8004/health|crypto_module (8004)"
    "http://127.0.0.1:8000/risk/health|risk_service (8000)"
    "http://127.0.0.1:8083/confidentiality/health|confiability_service (8083)"
    "http://127.0.0.1:8088/api/v1/health|classification_agent (8088)"
    "http://127.0.0.1:9009/docs|rl_engine (9009)"
)
for entry in "${healths[@]}"; do
    url="${entry%%|*}"; nome="${entry##*|}"
    if curl -sf -o /dev/null --max-time 10 "$url" 2>/dev/null; then ok "$nome"; else warn "$nome fora do ar"; fi
done

# --------------------------------------------------------------- 9. Resume
step "[9/9] INSTALACAO CONCLUIDA"
echo
echo "  Orquestrador:  http://127.0.0.1:8090"
echo "  Grafana:       http://127.0.0.1:3001"
echo "  RabbitMQ mgmt: http://127.0.0.1:15673 (user: qopsec)"
echo "  Segredos:      $SECRETS_FILE"
echo "  Iniciar de novo:  bash $INSTALL_DIR/start_qopsec_linux.sh"
echo "  Parar:            bash $INSTALL_DIR/stop_qopsec_linux.sh"
echo
echo "  Se o Docker foi configurado agora, pode ser preciso 'newgrp docker' "
echo "  ou reabrir o terminal para o usuario usar docker sem sudo."

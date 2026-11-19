#!/bin/bash
# Configura o ambiente Java e Maven.
# Prioridade 1: instalacao portatil na home (JDK 21 + Maven 3.9.11), usada quando
# o ambiente ainda nao tem os pacotes do sistema (ex.: maquina restaurada).
# Prioridade 2: instalacao do sistema (/usr/lib/jvm/java-21-openjdk-amd64 + /usr/share/maven).
PORTABLE_JAVA_HOME="/home/umbrel/.local/opt/jdk-21.0.13+9"
PORTABLE_MAVEN_HOME="/home/umbrel/.local/opt/apache-maven-3.9.11"
if [ -d "$PORTABLE_JAVA_HOME" ] && [ -d "$PORTABLE_MAVEN_HOME" ]; then
    export JAVA_HOME="$PORTABLE_JAVA_HOME"
    export MAVEN_HOME="$PORTABLE_MAVEN_HOME"
else
    export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"
    export MAVEN_HOME="${MAVEN_HOME:-/usr/share/maven}"
fi
export PATH="$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH"

cd /home/umbrel/projetos/Q-OPSEC
source /home/umbrel/projetos/Q-OPSEC/qopsec_env/bin/activate
echo "Atualizando dependências do Orquestrador..."
pip install -r requirements.txt -r requirements-architecture.txt --quiet

# Secrets são lidos apenas de arquivo protegido, nunca versionados.
SECRETS_FILE="/home/umbrel/.config/qopsec/middleware.env"
if [ -f "$SECRETS_FILE" ]; then
    set -a
    source "$SECRETS_FILE"
    set +a
fi

if [ -z "$QOPSEC_REDIS_PASSWORD" ] || [ -z "$QOPSEC_RABBITMQ_USER" ] || [ -z "$QOPSEC_RABBITMQ_PASSWORD" ]; then
    echo "ERRO: configure QOPSEC_REDIS_PASSWORD, QOPSEC_RABBITMQ_USER e QOPSEC_RABBITMQ_PASSWORD em $SECRETS_FILE" >&2
    exit 1
fi

# Tracing OTLP para o coletor gerenciado (qopsec_otel_collector).
# Desligue com QOPSEC_TRACING_ENABLED=false no ambiente.
export QOPSEC_TRACING_ENABLED="${QOPSEC_TRACING_ENABLED:-true}"
export QOPSEC_OTLP_ENDPOINT="${QOPSEC_OTLP_ENDPOINT:-http://127.0.0.1:4318/v1/traces}"
exec python3 orchestrator_linux.py

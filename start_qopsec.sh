#!/bin/bash
# Configura o ambiente Java e Maven
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export MAVEN_HOME=/usr/share/maven
export PATH=$JAVA_HOME/bin:$MAVEN_HOME/bin:$PATH

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
exec python3 orchestrator_linux.py

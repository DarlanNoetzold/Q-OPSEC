#!/bin/bash
# Configura o ambiente Java (detectado automaticamente)
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

source /home/umbrel/projetos/Q-OPSEC/qopsec_env/bin/activate
cd /home/umbrel/projetos/Q-OPSEC
echo "Atualizando dependências do Orquestrador..."
pip install -r requirements.txt --quiet
exec python3 orchestrator_linux.py

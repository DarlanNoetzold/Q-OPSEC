#!/bin/bash
# check_and_install.sh - Verificação de dependências locais (Sem Token)

PROJECT_ROOT="/home/umbrel/projetos/Q-OPSEC"
VENV_PATH="$PROJECT_ROOT/qopsec_env"

echo "=== Verificando Requisitos Q-OPSEC ==="

# 1. Verificar Java
if ! command -v java &> /dev/null; then
    echo "[ERRO] Java não encontrado. Instale o OpenJDK 17."
else
    echo -n "Java: "
    java -version 2>&1 | head -n 1
fi

# 2. Verificar Maven
if ! command -v mvn &> /dev/null; then
    echo "[ERRO] Maven não encontrado. Instale com: sudo apt install maven"
else
    echo -n "Maven: "
    mvn -version | head -n 1
fi

# 3. Verificar Docker
if ! command -v docker &> /dev/null; then
    echo "[ERRO] Docker não encontrado."
else
    echo -n "Docker: "
    docker --version
fi

# 4. Verificar Ambiente Python
if [ ! -d "$VENV_PATH" ]; then
    echo "[AVISO] Virtualenv não encontrado em $VENV_PATH. Criando..."
    python3 -m venv "$VENV_PATH"
fi

# 5. Instalar dependências de todos os sub-módulos
echo "Limpando e instalando dependências dos submódulos..."
source "$VENV_PATH/bin/activate"

# Procura todos os requirements.txt e instala
find "$PROJECT_ROOT" -name "requirements.txt" -exec pip install -r {} \;

echo "=== Verificação Concluída ==="

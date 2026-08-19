#!/bin/bash
set -e

echo "=========================================================="
echo "   Q-OPSEC MASTER INSTALLER - UMREL/LINUX COMPLIANT"
echo "=========================================================="

# 1. Atualizar e Instalar Dependências de Sistema (APT)
echo "[1/5] Instalando dependências de sistema (Native Build Tools)..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    libssl-dev \
    python3-dev \
    python3-venv \
    openjdk-21-jdk \
    maven \
    docker.io \
    docker-compose-v2 \
    mosquitto \
    mosquitto-clients

# 2. Configurar Ambiente Python
echo "[2/5] Configurando ambiente virtual Python 3.13..."
cd /home/umbrel/projetos/Q-OPSEC
if [ ! -d "qopsec_env" ]; then
    python3 -m venv qopsec_env
fi
source qopsec_env/bin/activate
pip install --upgrade pip setuptools wheel

# 3. Instalar Dependências de IA e Ciência de Dados (Pesadas)
echo "[3/5] Instalando pacotes Python (Data Science & Quantum)..."
# Nota: Instalamos em ordem para evitar conflitos de dependencias de build
pip install numpy==2.5.1 pandas==3.0.5 scipy==1.18.0
pip install matplotlib seaborn scikit-learn==1.9.0
pip install torch xgboost catboost lightgbm
pip install qiskit==2.5.1 qiskit-aer==0.17.2

# 4. Compilar liboqs e oqs-python (Cripto Pós-Quântica)
echo "[4/5] Compilando suporte a Criptografia Pós-Quântica (OQS)..."
if [ ! -d "liboqs" ]; then
    git clone --branch main https://github.com/open-quantum-safe/liboqs.git
fi
cd liboqs
mkdir -p build && cd build
cmake -GNinja -DOQS_USE_OPENSSL=ON ..
ninja
sudo ninja install
cd ../..

# Instalar oqs-python (Wrapper)
cd kms_service/liboqs-python || cd /home/umbrel/projetos/Q-OPSEC/kms_service/liboqs-python
pip install .
cd /home/umbrel/projetos/Q-OPSEC

# 5. Instalar Requisitos de todos os Submódulos
echo "[5/5] Instalando requisitos residuais de submódulos..."
find . -name "requirements.txt" -exec pip install -r {} \;

echo "----------------------------------------------------------"
echo "INSTALAÇÃO CONCLUÍDA COM SUCESSO!"
echo "Para ativar o ambiente: source /home/umbrel/projetos/Q-OPSEC/qopsec_env/bin/activate"
echo "----------------------------------------------------------"

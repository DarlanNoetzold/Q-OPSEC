# quantum_gateway/gateway.py
import os
from base64 import b64encode
from quantum_gateway.qkd_interface import (
    generate_bb84_key,
    generate_e91_key,
    generate_cv_qkd_key,
    generate_mdi_qkd_key,
    generate_decoy_state_key,
    generate_sarg04_key,
    generate_di_qkd_key,
)

QKD_ALGORITHMS = {
    "QKD_BB84": generate_bb84_key,
    "QKD_E91": generate_e91_key,
    "QKD_CV": generate_cv_qkd_key,
    "QKD_MDI": generate_mdi_qkd_key,
    "QKD_DecoyState": generate_decoy_state_key,
    "QKD_SARG04": generate_sarg04_key,
    "QKD_DI": generate_di_qkd_key,
}

def generate_key_from_gateway(algorithm: str):
    qkd_enabled = os.getenv("QKD_AVAILABLE", "false").lower() == "true"
    qkd_enabled = 1
    if not qkd_enabled:
        print(f"[QKD Gateway] QKD_AVAILABLE != true (valor='{os.getenv('QKD_AVAILABLE')}'). Ignorando QKD.")
        return None, None

    if algorithm not in QKD_ALGORITHMS:
        print(f"[QKD Gateway] Algoritmo não mapeado: {algorithm}")
        return None, None

    try:
        key_material = QKD_ALGORITHMS[algorithm]()
        if not key_material:
            print(f"[QKD Gateway] Função retornou vazio para {algorithm}")
            return None, None
        return algorithm, b64encode(key_material).decode()
    except Exception as e:
        print(f"[QKD Gateway] Erro ao gerar chave com {algorithm}: {e}")
        return None, None
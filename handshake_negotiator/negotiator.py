import yaml
import uuid
from datetime import datetime, timedelta


# Carrega fallback policy
def load_policies(path: str = "policies.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Mock do Hardware Detector (poderá ser API externa depois!)
def is_quantum_available() -> bool:
    # Aqui simulamos indisponibilidade 50% das vezes
    import random
    return random.random() > 0.5


def negotiate_algorithms(source_proposed, destination_proposed):
    """
    Negocia algoritmo comum entre source/destination + aplica fallback se necessário
    """
    policies = load_policies()

    # Interseção entre listas
    common = [alg for alg in source_proposed if alg in destination_proposed]

    if not common:
        # Se não há interseção, cai no default
        return policies["fallback"]["default"], True

    chosen = common[0]  # pega o primeiro em ordem de prioridade do source

    # Simula hardware detector caso escolha QKD
    if chosen.startswith("QKD") and not is_quantum_available():
        chosen = policies["fallback"]["if_qkd_unavailable"]
        return chosen, True

    # Caso PQC não disponível futuramente
    if chosen.startswith("Kyber") and False:  # simulação
        chosen = policies["fallback"]["if_pqc_unavailable"]
        return chosen, True

    return chosen, False


def create_session(alg: str, ttl_seconds: int = 60):
    """
    Cria sessão com ID único e validade
    """
    session_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
    return session_id, expires_at
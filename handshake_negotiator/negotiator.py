import yaml
import uuid
import random
from datetime import datetime, timedelta


def load_policies(path: str = "policies.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def is_quantum_available() -> bool:
    return random.random() > 0.4  # ~60% disponível


def negotiate_algorithms(src_props, dst_props):
    policies = load_policies()
    priority = policies.get("priority", [])

    # interseção entre os algoritmos propostos
    common = [alg for alg in priority if alg in src_props and alg in dst_props]

    if not common:
        return policies["fallback"]["default"], True

    chosen = common[0]

    # Teste de disponibilidade quântica
    if chosen.startswith("QKD") and not is_quantum_available():
        return policies["fallback"]["if_qkd_unavailable"], True

    # Se PQC ainda não estiver disponível (simulação futura)
    if chosen.startswith(("Kyber", "Dilithium", "Falcon", "Sphincs")) and False:
        return policies["fallback"]["if_pqc_unavailable"], True

    session_id = str(uuid.uuid4())
    return chosen, session_id


def create_session(alg: str, ttl_seconds: int = 300):
    session_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
    return session_id, expires_at
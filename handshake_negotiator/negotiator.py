import yaml
import uuid
import random
from datetime import datetime, timedelta
from models import NegotiationRequest


def load_policies(path: str = "policies.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def is_quantum_available() -> bool:
    return random.random() > 0.4  # ~60% disponível


def negotiate_algorithms(req: NegotiationRequest):
    proposed = req.proposed
    dst_props = req.dst_props or {}
    dst_algorithms = dst_props.get("algorithms", proposed)

    policies = load_policies()
    priority = policies.get("priority", [])

    common = [alg for alg in priority if alg in proposed and alg in dst_algorithms]

    if not common:
        return policies["fallback"]["default"], str(uuid.uuid4())

    chosen = common[0]

    if chosen.startswith("QKD") and not is_quantum_available():
        return policies["fallback"]["if_qkd_unavailable"], str(uuid.uuid4())

    if chosen.startswith(("Kyber", "Dilithium", "Falcon", "Sphincs")) and False:
        return policies["fallback"]["if_pqc_unavailable"], str(uuid.uuid4())

    session_id = str(uuid.uuid4())
    return chosen, session_id

def create_session(alg: str, ttl_seconds: int = 300):
    session_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
    return session_id, expires_at
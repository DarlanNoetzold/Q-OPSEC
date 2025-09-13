import yaml
import uuid
import random
from datetime import datetime, timedelta
from typing import Tuple
from models import NegotiationRequest

def load_policies(path: str = "policies.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def is_quantum_available() -> bool:
    return random.random() > 0.4  # ~60% disponível

def negotiate_algorithms(req: NegotiationRequest) -> Tuple[str, str, bool, str | None]:
    """
    Negotiate algorithms based on policies.

    Returns:
        (chosen_algorithm, session_id, fallback_applied, fallback_reason)
    """
    proposed = req.proposed
    dst_props = req.dst_props or {}
    dst_algorithms = dst_props.get("algorithms", proposed)

    policies = load_policies()
    priority = policies.get("priority", [])

    # Find common algorithms in priority order
    common = [alg for alg in priority if alg in proposed and alg in dst_algorithms]

    session_id = str(uuid.uuid4())

    if not common:
        fallback_alg = policies["fallback"]["default"]
        return fallback_alg, session_id, True, "NO_COMMON_ALGORITHMS"

    chosen = common[0]

    # Check QKD availability
    if chosen.startswith("QKD") and not is_quantum_available():
        chosen = policies["fallback"]["if_qkd_unavailable"]
        return chosen, session_id, True, "QKD_UNAVAILABLE"

    # Check PQC availability (placeholder desativado)
    if chosen.startswith(("Kyber", "Dilithium", "Falcon", "Sphincs")) and False:
        chosen = policies["fallback"]["if_pqc_unavailable"]
        return chosen, session_id, True, "PQC_UNAVAILABLE"

    # No fallback needed
    return chosen, session_id, False, None

def create_session(alg: str, ttl_seconds: int = 300):
    session_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
    return session_id, expires_at
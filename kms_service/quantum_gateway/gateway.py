from .hardware_detector import is_qkd_available
from .fallback_manager import choose_fallback
from .qkd_interface import qkd_bb84, qkd_e91
from .compatibility_layer import to_session_key

def generate_key_from_gateway(algorithm: str):
    if algorithm.startswith("QKD"):
        if not is_qkd_available():
            chosen = choose_fallback(algorithm)
            return chosen, None

        if algorithm == "QKD_BB84":
            return algorithm, to_session_key(qkd_bb84())
        if algorithm == "QKD_E91":
            return algorithm, to_session_key(qkd_e91())

    # Outras QKD families podem ser plugadas (CV-QKD, MDI-QKD, SARG04…)
    return algorithm, None
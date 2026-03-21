import os

def is_qkd_available() -> bool:
    """
    Detecta se hardware QKD está disponível.
    Futuro: integrar driver/hardware real (ex: QuNetSim ou IDQuantique SDK).
    """
    return os.getenv("QKD_AVAILABLE", "false").lower() == "true"
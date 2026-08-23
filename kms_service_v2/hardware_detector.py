import os

def is_qkd_available() -> bool:
    return os.getenv("QKD_AVAILABLE", "true").lower() == "true"

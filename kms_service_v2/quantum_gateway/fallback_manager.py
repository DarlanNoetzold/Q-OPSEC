FALLBACK_POLICIES = {
    "QKD_BB84": "Kyber1024",
    "QKD_E91": "Kyber1024",
    "QKD_CV": "Kyber768",
    "QKD_MDI": "Kyber1024",
    "QKD_SARG04": "Kyber768",
    "default": "AES256_GCM"
}

def choose_fallback(algorithm: str) -> str:
    return FALLBACK_POLICIES.get(algorithm, FALLBACK_POLICIES["default"])
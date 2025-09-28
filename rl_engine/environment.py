from typing import Dict, Any

ACTIONS = {
    0: "BB84",
    1: "E91",
    2: "CV-QKD",
    3: "Fallback-AES",
    4: "PostQuantum-RSA"
}

def extract_state(context: Dict[str, Any]) -> str:
    sec_level = context.get("security_level", "Low")
    risk = round(float(context.get("risk_score", 0.0)), 1)
    conf = round(float(context.get("conf_score", 0.0)), 1)
    return f"{sec_level}|{risk}|{conf}"
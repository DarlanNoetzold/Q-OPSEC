import os

import requests

HANDSHAKE_TIMEOUT = float(os.getenv("RL_HANDSHAKE_TIMEOUT", "60"))
_SESSION = requests.Session()


def send_to_handshake(handshake_url: str, payload: dict, timeout: float = HANDSHAKE_TIMEOUT):
    try:
        resp = _SESSION.post(handshake_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "failed", "error": str(e)}

import requests

def send_to_handshake(handshake_url: str, payload: dict):
    try:
        resp = requests.post(handshake_url, json=payload)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "failed", "error": str(e)}
import base64

def to_session_key(raw_bytes: bytes) -> str:
    return base64.b64encode(raw_bytes).decode()
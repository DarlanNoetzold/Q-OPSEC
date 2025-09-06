import os, base64
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend


def derive_key(material: bytes, length: int = 32, info: bytes = b"session-key") -> bytes:
    """
    Deriva uma chave simétrica fixa (AES-256) a partir de material arbitrário.

    Args:
        material (bytes): material bruto (ex: chave Kyber/QKD/raw bits).
        length (int): tamanho da chave final (bytes). Default: 32 (AES-256).
        info (bytes): contexto da derivação (default: b'session-key').

    Returns:
        bytes: chave derivada (tamanho fixo).
    """
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=info,
        backend=default_backend()
    )
    return hkdf.derive(material)


def generate_random_key() -> str:
    """ Gera uma chave AES-256 aleatória em base64 """
    raw = os.urandom(32)
    return base64.b64encode(raw).decode("utf-8")
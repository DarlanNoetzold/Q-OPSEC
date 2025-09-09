from crypto_utils import derive_key
from qunetsim.components import Host, Network
from qunetsim.objects import Logger

# Exemplo usando QuNetSim para simular QKD real
# https://github.com/QuNetSim/QuNetSim

def qkd_bb84() -> bytes:
    """
    Realiza BB84 via QuNetSim.
    """
    raw_bits = b"110010101..."
    return derive_key(raw_bits.encode(), 32)


def qkd_e91() -> bytes:
    """
    Realiza E91 (emaranhamento) via QuNetSim / hardware real
    """
    raw_bits = b"011101010..."
    return derive_key(raw_bits.encode(), 32)
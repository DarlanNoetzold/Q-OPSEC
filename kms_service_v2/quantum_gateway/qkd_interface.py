"""
QKD Interface - Quantum Key Distribution Protocol Simulations

Each protocol returns raw key material as bytes (pseudo-random).
"""

import os

def _random_key(bits: int = 256) -> bytes:
    """Generate pseudo-random key material of given size in bits."""
    return os.urandom(bits // 8)

def qkd_bb84_simulation() -> bytes: return _random_key()
def qkd_e91_simulation() -> bytes: return _random_key()
def qkd_cv_simulation() -> bytes: return _random_key()
def qkd_mdi_simulation() -> bytes: return _random_key()
def qkd_sarg04_simulation() -> bytes: return _random_key()
def qkd_decoy_state_simulation() -> bytes: return _random_key()
def qkd_device_independent_simulation() -> bytes: return _random_key()
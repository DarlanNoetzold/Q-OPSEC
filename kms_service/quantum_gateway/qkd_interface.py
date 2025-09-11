import os
from quantum_gateway.compatibility_layer import to_session_key

def _simulate_qkd_key(size: int = 32):
    return os.urandom(size)

def generate_bb84_key():
    return to_session_key(_simulate_qkd_key())

def generate_e91_key():
    return to_session_key(_simulate_qkd_key())

def generate_cv_qkd_key():
    return to_session_key(_simulate_qkd_key())

def generate_mdi_qkd_key():
    return to_session_key(_simulate_qkd_key())

def generate_decoy_state_key():
    return to_session_key(_simulate_qkd_key())

def generate_sarg04_key():
    return to_session_key(_simulate_qkd_key())

def generate_di_qkd_key():
    return to_session_key(_simulate_qkd_key())
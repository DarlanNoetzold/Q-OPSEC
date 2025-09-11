import base64
import os
import time
from typing import Optional, Tuple
import uuid
from crypto_utils import derive_key
from quantum_gateway.gateway import generate_key_from_gateway
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.primitives import serialization

# ====
# liboqs (Open Quantum Safe) - principal
# ====
OQS_AVAILABLE = False
try:
    import oqs

    OQS_AVAILABLE = True
    print("liboqs-python carregado com sucesso")
except ImportError:
    oqs = None
    OQS_AVAILABLE = False
    print("liboqs-python não disponível")

# ====
# pqcrypto fallback (se disponível)
# ====
PQC_AVAILABLE = False
try:
    import pqcrypto

    # Verifica se tem KEMs disponíveis
    try:
        from pqcrypto.kem import ntruhrss701

        PQC_AVAILABLE = True
        print("pqcrypto carregado com sucesso")
    except ImportError:
        print("pqcrypto instalado mas sem KEMs disponíveis")
        PQC_AVAILABLE = False
except ImportError:
    print("pqcrypto não disponível")
    PQC_AVAILABLE = False


def _normalize_oqs_algorithm(requested: str, kem_mechs: list, sig_mechs: list) -> Tuple[Optional[str], str]:
    """
    Tenta mapear o nome solicitado para um mecanismo disponível pelo OQS.
    Retorna (normalized_name_ou_None, category: 'kem'|'sig'|'')
    """
    if not requested:
        return None, ""

    # Match exato primeiro
    if requested in kem_mechs:
        return requested, "kem"
    if requested in sig_mechs:
        return requested, "sig"

    # Aliases comuns (pós-NIST)
    aliases = {
        # Kyber -> ML-KEM
        "Kyber512": "ML-KEM-512",
        "Kyber768": "ML-KEM-768",
        "Kyber1024": "ML-KEM-1024",

        # Dilithium -> ML-DSA
        "Dilithium2": "ML-DSA-44",
        "Dilithium3": "ML-DSA-65",
        "Dilithium5": "ML-DSA-87",

        # Falcon variants
        "Falcon-512": "Falcon-512",
        "Falcon-1024": "Falcon-1024",
        "FALCON-512": "Falcon-512",
        "FALCON-1024": "Falcon-1024",
    }

    candidate = aliases.get(requested)
    if candidate:
        if candidate in kem_mechs:
            return candidate, "kem"
        if candidate in sig_mechs:
            return candidate, "sig"

    # Tentativa mais tolerante: ignora caixa e hífens
    def normalize(s: str) -> str:
        return s.replace("-", "").replace("_", "").upper()

    rq = normalize(requested)
    for m in kem_mechs:
        if normalize(m) == rq:
            return m, "kem"
    for m in sig_mechs:
        if normalize(m) == rq:
            return m, "sig"

    return None, ""


def _oqs_kem_mechanisms():
    """Detecta KEMs disponíveis no liboqs (compatibilidade multi-versão)"""
    if not OQS_AVAILABLE:
        return []

    for fn_name in [
        "get_supported_KEM_mechanisms",
        "get_available_KEM_mechanisms",
        "get_enabled_KEM_mechanisms",
        "get_enabled_KEMs"
    ]:
        fn = getattr(oqs, fn_name, None)
        if fn:
            try:
                mechs = fn()
                if isinstance(mechs, (list, tuple)):
                    return list(mechs)
            except Exception:
                continue
    return []


def _oqs_sig_mechanisms():
    """Detecta Signatures disponíveis no liboqs (compatibilidade multi-versão)"""
    if not OQS_AVAILABLE:
        return []

    for fn_name in [
        "get_supported_sig_mechanisms",
        "get_available_sig_mechanisms",
        "get_enabled_sig_mechanisms",
        "get_enabled_Sigs"
    ]:
        fn = getattr(oqs, fn_name, None)
        if fn:
            try:
                mechs = fn()
                if isinstance(mechs, (list, tuple)):
                    return list(mechs)
            except Exception:
                continue
    return []


def _get_pqcrypto_kems():
    """Detecta KEMs realmente disponíveis no pqcrypto"""
    if not PQC_AVAILABLE:
        return []

    available_kems = []
    kem_modules = [
        ('ntruhrss701', 'NTRU-HRSS-701'),
        ('ntruhps2048509', 'NTRU-HPS-2048-509'),
        ('ntruhps2048677', 'NTRU-HPS-2048-677'),
        ('lightsaber', 'LightSaber'),
        ('saber', 'Saber'),
        ('firesaber', 'FireSaber'),
        ('frodokem640aes', 'FrodoKEM-640-AES'),
        ('frodokem976aes', 'FrodoKEM-976-AES'),
        ('frodokem1344aes', 'FrodoKEM-1344-AES')
    ]

    for module_name, algo_name in kem_modules:
        try:
            __import__(f'pqcrypto.kem.{module_name}')
            available_kems.append(algo_name)
        except ImportError:
            continue

    return available_kems


def _get_pqcrypto_sigs():
    """Detecta Signatures realmente disponíveis no pqcrypto"""
    if not PQC_AVAILABLE:
        return []

    available_sigs = []
    sig_modules = [
        ('dilithium2', 'Dilithium2'),
        ('dilithium3', 'Dilithium3'),
        ('dilithium5', 'Dilithium5'),
        ('falcon512', 'Falcon-512'),
        ('falcon1024', 'Falcon-1024'),
        ('sphincssha256128ssimple', 'SPHINCS+-SHA256-128s'),
        ('sphincssha256192ssimple', 'SPHINCS+-SHA256-192s'),
        ('sphincssha256256ssimple', 'SPHINCS+-SHA256-256s')
    ]

    for module_name, algo_name in sig_modules:
        try:
            __import__(f'pqcrypto.sign.{module_name}')
            available_sigs.append(algo_name)
        except ImportError:
            continue

    return available_sigs


def generate_key(algorithm: str) -> Tuple[str, str]:
    """
    Gera chave de sessão segura via:
    1. QKD (Quantum Gateway)
    2. PQC (liboqs - principal)
    3. PQC (pqcrypto - fallback)
    4. Clássicos (AES, ChaCha, RSA, ECC)

    Retorna: (algorithm_used, key_material_base64)
    """

    # ====
    # 1. QKD via Quantum Gateway
    # ====
    chosen_algo, key_material = generate_key_from_gateway(algorithm)
    if key_material:
        return chosen_algo, key_material

    # ====
    # 2. PQC via liboqs (principal)
    # ====
    if OQS_AVAILABLE:
        kem_mechs = _oqs_kem_mechanisms()
        sig_mechs = _oqs_sig_mechanisms()

        normalized, cat = _normalize_oqs_algorithm(algorithm, kem_mechs, sig_mechs)
        if normalized and cat == "kem":
            try:
                with oqs.KeyEncapsulation(normalized) as kem:
                    public_key = kem.generate_keypair()
                    ciphertext, session_key = kem.encap_secret(public_key)
                    return normalized, base64.b64encode(session_key).decode()
            except Exception as e:
                print(f"Erro ao gerar KEM {normalized} via liboqs: {e}")

        if normalized and cat == "sig":
            try:
                with oqs.Signature(normalized) as sig:
                    public_key = sig.generate_keypair()
                    return normalized, base64.b64encode(derive_key(public_key, 32)).decode()
            except Exception as e:
                print(f"Erro ao gerar Signature {normalized} via liboqs: {e}")

    # ====
    # 3. PQC via pqcrypto (fallback) - implementação mais robusta
    # ====
    if PQC_AVAILABLE:
        pqc_kem_map = {
            "NTRU-HRSS-701": ("pqcrypto.kem.ntruhrss701", "ntruhrss701"),
            "NTRU-HPS-2048-509": ("pqcrypto.kem.ntruhps2048509", "ntruhps2048509"),
            "NTRU-HPS-2048-677": ("pqcrypto.kem.ntruhps2048677", "ntruhps2048677"),
            "LightSaber": ("pqcrypto.kem.lightsaber", "lightsaber"),
            "Saber": ("pqcrypto.kem.saber", "saber"),
            "FireSaber": ("pqcrypto.kem.firesaber", "firesaber"),
            "FrodoKEM-640-AES": ("pqcrypto.kem.frodokem640aes", "frodokem640aes"),
            "FrodoKEM-976-AES": ("pqcrypto.kem.frodokem976aes", "frodokem976aes"),
            "FrodoKEM-1344-AES": ("pqcrypto.kem.frodokem1344aes", "frodokem1344aes")
        }

        pqc_sig_map = {
            "Dilithium2": ("pqcrypto.sign.dilithium2", "dilithium2"),
            "Dilithium3": ("pqcrypto.sign.dilithium3", "dilithium3"),
            "Dilithium5": ("pqcrypto.sign.dilithium5", "dilithium5"),
            "Falcon-512": ("pqcrypto.sign.falcon512", "falcon512"),
            "Falcon-1024": ("pqcrypto.sign.falcon1024", "falcon1024"),
            "SPHINCS+-SHA256-128s": ("pqcrypto.sign.sphincssha256128ssimple", "sphincssha256128ssimple"),
            "SPHINCS+-SHA256-192s": ("pqcrypto.sign.sphincssha256192ssimple", "sphincssha256192ssimple"),
            "SPHINCS+-SHA256-256s": ("pqcrypto.sign.sphincssha256256ssimple", "sphincssha256256ssimple")
        }

        # Tenta KEMs
        if algorithm in pqc_kem_map:
            module_path, module_name = pqc_kem_map[algorithm]
            try:
                module = __import__(module_path, fromlist=[module_name])
                pk, sk = module.generate_keypair()
                ct, ss = module.encapsulate(pk)
                return algorithm, base64.b64encode(ss).decode()
            except ImportError:
                pass

        # Tenta Signatures
        if algorithm in pqc_sig_map:
            module_path, module_name = pqc_sig_map[algorithm]
            try:
                module = __import__(module_path, fromlist=[module_name])
                pk, sk = module.generate_keypair()
                return algorithm, base64.b64encode(derive_key(pk, 32)).decode()
            except ImportError:
                pass

    # ====
    # 4. Clássicos (sempre disponíveis)
    # ====
    classical_algorithms = {
        # AES family
        "AES256_GCM": lambda: AESGCM.generate_key(bit_length=256),
        "AES128_GCM": lambda: AESGCM.generate_key(bit_length=128),

        # ChaCha20-Poly1305
        "ChaCha20_Poly1305": lambda: ChaCha20Poly1305.generate_key(),

        # Legacy (não recomendados)
        "3DES": lambda: os.urandom(24),  # 3DES usa 192 bits
        "Blowfish": lambda: os.urandom(32),  # Blowfish aceita até 448 bits
    }

    if algorithm in classical_algorithms:
        key = classical_algorithms[algorithm]()
        return algorithm, base64.b64encode(key).decode()

    # RSA (deriva material da chave privada)
    rsa_algorithms = {
        "RSA2048": 2048,
        "RSA4096": 4096
    }

    if algorithm in rsa_algorithms:
        key_size = rsa_algorithms[algorithm]
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        return algorithm, base64.b64encode(derive_key(key_bytes, 32)).decode()

    # ECC (Elliptic Curve Cryptography)
    ecc_algorithms = {
        "ECDH_P256": ec.SECP256R1(),
        "ECDH_P384": ec.SECP384R1(),
        "ECDH_Curve25519": ec.X25519()
    }

    if algorithm in ecc_algorithms:
        curve = ecc_algorithms[algorithm]
        private_key = ec.generate_private_key(curve)

        if algorithm == "ECDH_Curve25519":
            key_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            key_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        return algorithm, base64.b64encode(derive_key(key_bytes, 32)).decode()

    # Fallback para QKD quando não disponível
    if algorithm.startswith("QKD"):
        print(f"[KMS] QKD solicitado ({algorithm}) mas não gerou material. Aplicando fallback para AES256_GCM.")
        key = AESGCM.generate_key(bit_length=256)
        return "AES256_GCM", base64.b64encode(key).decode()

    # Se chegou aqui, algoritmo não suportado
    raise ValueError(f"Algoritmo '{algorithm}' não suportado pelo KMS")


def build_session(session_id: Optional[str], algorithm: str, ttl_seconds: int) -> Tuple[
    str, str, str, int, bool, Optional[str], str]:
    """
    Constrói uma sessão de chave completa.

    Returns:
        Tuple com: (session_id, selected_algorithm, key_material, expires_at,
                   fallback_applied, fallback_reason, source_of_key)
    """
    # Gera chave
    selected_alg, key_material = generate_key(algorithm)

    # Detecta fallback
    fallback_applied = selected_alg != algorithm
    fallback_reason = None
    if fallback_applied:
        if algorithm.startswith("QKD"):
            fallback_reason = "QKD_UNAVAILABLE"
        elif algorithm.upper().startswith(
                ("KYBER", "ML-KEM", "DILITHIUM", "FALCON", "SPHINCS", "NTRU", "SABER", "FRODO")):
            fallback_reason = "PQC_UNAVAILABLE"
        else:
            fallback_reason = "ALGO_NOT_SUPPORTED"

    # Inferir fonte da chave para telemetria
    if selected_alg.startswith("QKD"):
        source_of_key = "qkd"
    elif selected_alg.upper().startswith(
            ("KYBER", "ML-KEM", "DILITHIUM", "FALCON", "SPHINCS", "NTRU", "SABER", "FRODO")):
        source_of_key = "pqc"
    else:
        source_of_key = "classical"

    # Calcula expiração
    expires_at = int(time.time()) + int(ttl_seconds)
    sid = session_id or str(uuid.uuid4())

    return sid, selected_alg, key_material, expires_at, fallback_applied, fallback_reason, source_of_key


def get_supported_algorithms():
    """
    Retorna lista de todos os algoritmos suportados pelo KMS.
    """
    supported = {
        "classical": [
            "AES256_GCM", "AES128_GCM", "ChaCha20_Poly1305",
            "RSA2048", "RSA4096",
            "ECDH_P256", "ECDH_P384", "ECDH_Curve25519",
            "3DES", "Blowfish"
        ],
        "pqc_kems": [],
        "pqc_signatures": [],
        "qkd": [],
        "oqs_kems": [],
        "oqs_signatures": []
    }

    # liboqs (principal)
    if OQS_AVAILABLE:
        supported["oqs_kems"] = _oqs_kem_mechanisms()
        supported["oqs_signatures"] = _oqs_sig_mechanisms()

    # pqcrypto (fallback)
    if PQC_AVAILABLE:
        supported["pqc_kems"] = _get_pqcrypto_kems()
        supported["pqc_signatures"] = _get_pqcrypto_sigs()

    # QKD via Quantum Gateway
    qkd_available = os.getenv("QKD_AVAILABLE", "false").lower() == "true"
    if qkd_available:
        supported["qkd"] = [
            "QKD_BB84", "QKD_E91", "QKD_CV", "QKD_MDI",
            "QKD_SARG04", "QKD_DecoyState", "QKD_DI"
        ]

    return supported


def get_algorithm_info(algorithm: str):
    """
    Retorna informações sobre um algoritmo específico.
    """
    info = {
        "algorithm": algorithm,
        "category": "unknown",
        "security_level": "unknown",
        "key_size_bits": 256,  # padrão: 256 bits
        "quantum_resistant": False,
        "recommended": True
    }

    # Clássicos simétricos
    if algorithm in ["AES256_GCM", "ChaCha20_Poly1305"]:
        info.update({
            "category": "symmetric",
            "security_level": "high",
            "quantum_resistant": False,
            "recommended": True
        })
    elif algorithm == "AES128_GCM":
        info.update({
            "category": "symmetric",
            "security_level": "medium",
            "key_size_bits": 128,
            "quantum_resistant": False,
            "recommended": True
        })

    # Clássicos assimétricos
    elif algorithm.startswith("RSA"):
        key_size = int(algorithm.replace("RSA", ""))
        info.update({
            "category": "asymmetric",
            "security_level": "high" if key_size >= 4096 else "medium",
            "key_size_bits": key_size,
            "quantum_resistant": False,
            "recommended": key_size >= 2048
        })
    elif algorithm.startswith("ECDH"):
        info.update({
            "category": "asymmetric",
            "security_level": "high",
            "quantum_resistant": False,
            "recommended": True
        })

    # PQC KEMs
    elif algorithm.startswith(("Kyber", "ML-KEM", "NTRU", "Saber", "Frodo")):
        info.update({
            "category": "pqc_kem",
            "security_level": "high",
            "quantum_resistant": True,
            "recommended": True
        })

    # PQC Signatures
    elif algorithm.startswith(("Dilithium", "ML-DSA", "Falcon", "SPHINCS")):
        info.update({
            "category": "pqc_signature",
            "security_level": "high",
            "quantum_resistant": True,
            "recommended": True
        })

    # QKD
    elif algorithm.startswith("QKD"):
        info.update({
            "category": "qkd",
            "security_level": "maximum",
            "quantum_resistant": True,
            "recommended": True
        })

    # Legacy (não recomendados)
    elif algorithm in ["3DES", "Blowfish"]:
        info.update({
            "category": "legacy",
            "security_level": "low",
            "quantum_resistant": False,
            "recommended": False
        })

    return info
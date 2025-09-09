import base64
import os
from datetime import datetime, timedelta
from crypto_utils import derive_key
from quantum_gateway.gateway import generate_key_from_gateway
from quantum_gateway.compatibility_layer import to_session_key
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.asymmetric import rsa, ec
from cryptography.hazmat.primitives import serialization

# ========================================
# liboqs (Open Quantum Safe) - principal
# ========================================
OQS_AVAILABLE = False
try:
    import oqs

    OQS_AVAILABLE = True
    print("liboqs-python carregado com sucesso")
except ImportError:
    oqs = None
    OQS_AVAILABLE = False
    print("liboqs-python não disponível")

# ========================================
# pqcrypto fallback (se disponível)
# ========================================
PQC_AVAILABLE = False
try:
    # Tenta importar apenas os módulos que realmente existem
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

def _normalize_oqs_algorithm(requested: str, kem_mechs: list[str], sig_mechs: list[str]) -> tuple[str | None, str]:
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

        # Alguns pacotes expõem falcon em caixa alta, outros com hífen
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


def generate_key(algorithm: str):
    """
    Gera chave de sessão segura via:
    1. QKD (Quantum Gateway)
    2. PQC (liboqs - principal)
    3. PQC (pqcrypto - fallback)
    4. Clássicos (AES, ChaCha, RSA, ECC)

    Retorna: (algorithm_used, key_material_base64)
    """

    # ========================================
    # 1. QKD via Quantum Gateway
    # ========================================
    chosen_algo, key_material = generate_key_from_gateway(algorithm)
    if key_material:
        return chosen_algo, key_material

    # ========================================
    # 2. PQC via liboqs (principal)
    # ========================================
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

    # ========================================
    # 3. PQC via pqcrypto (fallback)
    # ========================================
    if PQC_AVAILABLE:
        # KEMs - NTRU family
        if algorithm == "NTRU-HRSS-701":
            try:
                from pqcrypto.kem import ntruhrss701
                pk, sk = ntruhrss701.generate_keypair()
                ct, ss = ntruhrss701.encapsulate(pk)
                return algorithm, base64.b64encode(ss).decode()
            except ImportError:
                pass

        if algorithm == "NTRU-HPS-2048-509":
            try:
                from pqcrypto.kem import ntruhps2048509
                pk, sk = ntruhps2048509.generate_keypair()
                ct, ss = ntruhps2048509.encapsulate(pk)
                return algorithm, base64.b64encode(ss).decode()
            except ImportError:
                pass

        if algorithm == "NTRU-HPS-2048-677":
            try:
                from pqcrypto.kem import ntruhps2048677
                pk, sk = ntruhps2048677.generate_keypair()
                ct, ss = ntruhps2048677.encapsulate(pk)
                return algorithm, base64.b64encode(ss).decode()
            except ImportError:
                pass

        # KEMs - SABER family
        if algorithm == "LightSaber":
            try:
                from pqcrypto.kem import lightsaber
                pk, sk = lightsaber.generate_keypair()
                ct, ss = lightsaber.encapsulate(pk)
                return algorithm, base64.b64encode(ss).decode()
            except ImportError:
                pass

        if algorithm == "Saber":
            try:
                from pqcrypto.kem import saber
                pk, sk = saber.generate_keypair()
                ct, ss = saber.encapsulate(pk)
                return algorithm, base64.b64encode(ss).decode()
            except ImportError:
                pass

        if algorithm == "FireSaber":
            try:
                from pqcrypto.kem import firesaber
                pk, sk = firesaber.generate_keypair()
                ct, ss = firesaber.encapsulate(pk)
                return algorithm, base64.b64encode(ss).decode()
            except ImportError:
                pass

        # KEMs - FrodoKEM family
        if algorithm == "FrodoKEM-640-AES":
            try:
                from pqcrypto.kem import frodokem640aes
                pk, sk = frodokem640aes.generate_keypair()
                ct, ss = frodokem640aes.encapsulate(pk)
                return algorithm, base64.b64encode(ss).decode()
            except ImportError:
                pass

        if algorithm == "FrodoKEM-976-AES":
            try:
                from pqcrypto.kem import frodokem976aes
                pk, sk = frodokem976aes.generate_keypair()
                ct, ss = frodokem976aes.encapsulate(pk)
                return algorithm, base64.b64encode(ss).decode()
            except ImportError:
                pass

        if algorithm == "FrodoKEM-1344-AES":
            try:
                from pqcrypto.kem import frodokem1344aes
                pk, sk = frodokem1344aes.generate_keypair()
                ct, ss = frodokem1344aes.encapsulate(pk)
                return algorithm, base64.b64encode(ss).decode()
            except ImportError:
                pass

        # Signatures - Dilithium family
        if algorithm == "Dilithium2":
            try:
                from pqcrypto.sign import dilithium2
                pk, sk = dilithium2.generate_keypair()
                return algorithm, base64.b64encode(derive_key(pk, 32)).decode()
            except ImportError:
                pass

        if algorithm == "Dilithium3":
            try:
                from pqcrypto.sign import dilithium3
                pk, sk = dilithium3.generate_keypair()
                return algorithm, base64.b64encode(derive_key(pk, 32)).decode()
            except ImportError:
                pass

        if algorithm == "Dilithium5":
            try:
                from pqcrypto.sign import dilithium5
                pk, sk = dilithium5.generate_keypair()
                return algorithm, base64.b64encode(derive_key(pk, 32)).decode()
            except ImportError:
                pass

        # Signatures - Falcon family
        if algorithm == "Falcon-512":
            try:
                from pqcrypto.sign import falcon512
                pk, sk = falcon512.generate_keypair()
                return algorithm, base64.b64encode(derive_key(pk, 32)).decode()
            except ImportError:
                pass

        if algorithm == "Falcon-1024":
            try:
                from pqcrypto.sign import falcon1024
                pk, sk = falcon1024.generate_keypair()
                return algorithm, base64.b64encode(derive_key(pk, 32)).decode()
            except ImportError:
                pass

        # Signatures - SPHINCS+ family
        if algorithm == "SPHINCS+-SHA256-128s":
            try:
                from pqcrypto.sign import sphincssha256128ssimple
                pk, sk = sphincssha256128ssimple.generate_keypair()
                return algorithm, base64.b64encode(derive_key(pk, 32)).decode()
            except ImportError:
                pass

        if algorithm == "SPHINCS+-SHA256-192s":
            try:
                from pqcrypto.sign import sphincssha256192ssimple
                pk, sk = sphincssha256192ssimple.generate_keypair()
                return algorithm, base64.b64encode(derive_key(pk, 32)).decode()
            except ImportError:
                pass

        if algorithm == "SPHINCS+-SHA256-256s":
            try:
                from pqcrypto.sign import sphincssha256256ssimple
                pk, sk = sphincssha256256ssimple.generate_keypair()
                return algorithm, base64.b64encode(derive_key(pk, 32)).decode()
            except ImportError:
                pass

    # ========================================
    # 4. Clássicos (sempre disponíveis)
    # ========================================

    # AES family
    if algorithm == "AES256_GCM":
        key = AESGCM.generate_key(bit_length=256)
        return algorithm, base64.b64encode(key).decode()

    if algorithm == "AES128_GCM":
        key = AESGCM.generate_key(bit_length=128)
        return algorithm, base64.b64encode(key).decode()

    # ChaCha20-Poly1305
    if algorithm == "ChaCha20_Poly1305":
        key = ChaCha20Poly1305.generate_key()
        return algorithm, base64.b64encode(key).decode()

    # RSA (deriva material da chave privada)
    if algorithm == "RSA2048":
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        return algorithm, base64.b64encode(derive_key(key_bytes, 32)).decode()

    if algorithm == "RSA4096":
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        return algorithm, base64.b64encode(derive_key(key_bytes, 32)).decode()

    # ECC (Elliptic Curve Cryptography)
    if algorithm == "ECDH_P256":
        private_key = ec.generate_private_key(ec.SECP256R1())
        key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        return algorithm, base64.b64encode(derive_key(key_bytes, 32)).decode()

    if algorithm == "ECDH_P384":
        private_key = ec.generate_private_key(ec.SECP384R1())
        key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        return algorithm, base64.b64encode(derive_key(key_bytes, 32)).decode()

    if algorithm == "ECDH_Curve25519":
        private_key = ec.generate_private_key(ec.X25519())
        key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )
        return algorithm, base64.b64encode(derive_key(key_bytes, 32)).decode()

    # 3DES (legacy, não recomendado)
    if algorithm == "3DES":
        key = os.urandom(24)  # 3DES usa 192 bits (24 bytes)
        return algorithm, base64.b64encode(derive_key(key, 32)).decode()

    # Blowfish (legacy)
    if algorithm == "Blowfish":
        key = os.urandom(32)  # Blowfish aceita chaves de 32-448 bits
        return algorithm, base64.b64encode(derive_key(key, 32)).decode()

    # Se chegou aqui, algoritmo não suportado
    raise ValueError(f"Algoritmo '{algorithm}' não suportado pelo KMS")


def build_session(session_id: str, algorithm: str, ttl: int = 300):
    """
    Constrói uma sessão completa com chave gerada.

    Args:
        session_id: ID único da sessão
        algorithm: Algoritmo para geração da chave
        ttl: Time-to-live em segundos

    Returns:
        tuple: (session_id, algorithm_used, key_material_base64, expires_datetime)
    """
    alg, key_material = generate_key(algorithm)
    expires = datetime.utcnow() + timedelta(seconds=ttl)
    return session_id, alg, key_material, expires


def get_supported_algorithms():
    """
    Retorna lista de todos os algoritmos suportados pelo KMS.

    Returns:
        dict: Categorias de algoritmos disponíveis
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

    Args:
        algorithm: Nome do algoritmo

    Returns:
        dict: Informações do algoritmo (categoria, segurança, etc.)
    """
    info = {
        "algorithm": algorithm,
        "category": "unknown",
        "security_level": "unknown",
        "key_size": "32",  # padrão: 256 bits
        "quantum_resistant": False
    }

    # Clássicos
    if algorithm in ["AES256_GCM", "ChaCha20_Poly1305"]:
        info.update({
            "category": "symmetric",
            "security_level": "high",
            "quantum_resistant": False
        })
    elif algorithm in ["AES128_GCM"]:
        info.update({
            "category": "symmetric",
            "security_level": "medium",
            "quantum_resistant": False
        })
    elif algorithm.startswith("RSA"):
        info.update({
            "category": "asymmetric",
            "security_level": "high" if "4096" in algorithm else "medium",
            "quantum_resistant": False
        })
    elif algorithm.startswith("ECDH"):
        info.update({
            "category": "asymmetric",
            "security_level": "high",
            "quantum_resistant": False
        })

    # PQC
    elif algorithm.startswith(("Kyber", "NTRU", "Saber", "Frodo")):
        info.update({
            "category": "pqc_kem",
            "security_level": "high",
            "quantum_resistant": True
        })
    elif algorithm.startswith(("Dilithium", "Falcon", "SPHINCS")):
        info.update({
            "category": "pqc_signature",
            "security_level": "high",
            "quantum_resistant": True
        })

    # QKD
    elif algorithm.startswith("QKD"):
        info.update({
            "category": "qkd",
            "security_level": "maximum",
            "quantum_resistant": True
        })

    return info
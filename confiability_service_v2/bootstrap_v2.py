"""
Bootstrap do Trust Engine V2
"""
from config.trust_config import TrustConfig
from core.orchestrator import TrustOrchestrator
from storage.trust_repository import TrustRepository
from storage.fingerprint_repo import FingerprintRepository

# Singleton instances
_orchestrator = None


def get_trust_orchestrator() -> TrustOrchestrator:
    """Retorna instância singleton do orchestrator"""
    global _orchestrator

    if _orchestrator is None:
        config = TrustConfig()
        trust_repo = TrustRepository()
        fingerprint_repo = FingerprintRepository()

        _orchestrator = TrustOrchestrator(
            config=config,
            trust_repo=trust_repo,
            fingerprint_repo=fingerprint_repo
        )

    return _orchestrator
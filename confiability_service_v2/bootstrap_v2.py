"""
Bootstrap do Trust Engine V2
Inicializa e configura todos os componentes do sistema
"""
from core.orchestrator import TrustOrchestrator
from config.trust_config import TrustConfig
from storage.trust_repository import TrustRepository
from storage.fingerprint_repo import FingerprintRepository

# Singleton global
_trust_orchestrator: TrustOrchestrator | None = None


def initialize_trust_engine() -> TrustOrchestrator:
    """
    Inicializa o Trust Engine V2 com todas as dependências
    Retorna a instância singleton do TrustOrchestrator
    """
    global _trust_orchestrator

    if _trust_orchestrator is not None:
        return _trust_orchestrator

    # 1. Configuração
    config = TrustConfig()

    # 2. Repositórios
    trust_repo = TrustRepository()
    fingerprint_repo = FingerprintRepository()

    # 3. Criar orchestrator (SEM trust_graph)
    _trust_orchestrator = TrustOrchestrator(
        config=config,
        trust_repo=trust_repo,
        fingerprint_repo=fingerprint_repo
    )

    print("✅ Trust Engine V2 initialized successfully")
    print(f"   - Signals loaded: {len(_trust_orchestrator.signals)}")
    print(f"   - Signal names: {[s.name for s in _trust_orchestrator.signals]}")

    return _trust_orchestrator


def get_trust_orchestrator() -> TrustOrchestrator:
    """
    Retorna a instância singleton do TrustOrchestrator
    Inicializa se ainda não foi inicializado
    """
    global _trust_orchestrator

    if _trust_orchestrator is None:
        return initialize_trust_engine()

    return _trust_orchestrator


def reset_trust_engine():
    """
    Reseta o Trust Engine (útil para testes)
    """
    global _trust_orchestrator
    _trust_orchestrator = None
"""
Base abstrata para Trust Signals
Define a interface comum para todos os signals
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from core.trust_context import TrustContext
from core.trust_result import SignalResult
from config.trust_config import TrustConfig
from storage.trust_repository import TrustRepository


class TrustSignal(ABC):
    """
    Classe base abstrata para todos os Trust Signals
    Cada signal implementa uma dimensão específica de avaliação de confiança
    """

    def __init__(self, config: TrustConfig, trust_repo: TrustRepository):
        """
        Inicializa o signal com configuração e repositório

        Args:
            config: Configuração global do Trust Engine
            trust_repo: Repositório de histórico de avaliações
        """
        self.config = config
        self.trust_repo = trust_repo

        # Carrega params específicos do signal da configuração
        signal_config = config.get_signal_config(self.name)
        self.params = signal_config.params if signal_config else {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome único do signal"""
        pass

    @abstractmethod
    def evaluate(self, context: TrustContext) -> SignalResult:
        """
        Avalia o contexto e retorna um resultado

        Args:
            context: Contexto da avaliação (payload + metadata)

        Returns:
            SignalResult com score, confidence e metadata
        """
        pass

    def _create_result(
        self,
        score: float,
        confidence: float = 1.0,
        metadata: Dict[str, Any] = None
    ) -> SignalResult:
        """
        Helper para criar SignalResult

        Args:
            score: Score normalizado [0, 1]
            confidence: Confiança na avaliação [0, 1]
            metadata: Metadados adicionais

        Returns:
            SignalResult
        """
        return SignalResult(
            signal_name=self.name,
            score=max(0.0, min(1.0, score)),  # Clamp [0, 1]
            confidence=max(0.0, min(1.0, confidence)),
            metadata=metadata or {}
        )

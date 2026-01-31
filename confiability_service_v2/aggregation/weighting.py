"""
Dynamic Weighting - Ajusta pesos dos signals baseado em contexto
"""
from typing import Dict, List
from core.trust_context import TrustContext
from core.trust_result import SignalResult
from config.trust_config import TrustConfig
from storage.trust_repository import TrustRepository


class DynamicWeighting:
    """
    Ajusta pesos dos signals dinamicamente baseado em:
    - Tipo de dado
    - Ambiente
    - Histórico da entidade
    - Performance dos signals
    """

    def __init__(self, config: TrustConfig, trust_repo: TrustRepository):
        """
        Inicializa dynamic weighting

        Args:
            config: Configuração do Trust Engine
            trust_repo: Repositório de histórico
        """
        self.config = config
        self.trust_repo = trust_repo

    def calculate_weights(
        self,
        context: TrustContext,
        signal_results: List[SignalResult]
    ) -> Dict[str, float]:
        """
        Calcula pesos ajustados para cada signal

        Args:
            context: Contexto da avaliação
            signal_results: Resultados dos signals

        Returns:
            Dict {signal_name: weight}
        """
        weights = {}

        for sr in signal_results:
            # Peso base da configuração (com fallback)
            signal_config = self.config.get_signal_config(sr.signal_name)
            base_weight = signal_config.weight if signal_config else 1.0

            # Ajustes contextuais
            weight = base_weight

            # 1. Ajuste por tipo de dado
            if context.data_type == "event":
                if sr.signal_name == "temporal":
                    weight *= 1.3  # Eventos são sensíveis a tempo
            elif context.data_type == "profile":
                if sr.signal_name == "semantic_consistency":
                    weight *= 1.2  # Perfis devem ser consistentes

            # 2. Ajuste por ambiente
            if context.environment == "production":
                if sr.signal_name in ["anomaly_detection", "source_reliability"]:
                    weight *= 1.2  # Mais rigoroso em produção

            # 3. Ajuste por confidence do signal
            weight *= sr.confidence

            # 4. Ajuste por histórico da entidade
            if context.entity_id:
                history = self.trust_repo.get_entity_history(context.entity_id, limit=10)
                if len(history) < 3:
                    # Entidade nova - aumenta peso de source_reliability
                    if sr.signal_name == "source_reliability":
                        weight *= 1.3

            weights[sr.signal_name] = max(0.1, weight)  # Peso mínimo 0.1

        # Normaliza pesos para somar 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


class AdaptiveWeighting:
    """
    Aprende pesos ótimos ao longo do tempo baseado em performance
    (Versão simplificada - pode ser expandida com ML)
    """

    def __init__(self, config: TrustConfig, trust_repo: TrustRepository):
        """
        Inicializa adaptive weighting

        Args:
            config: Configuração do Trust Engine
            trust_repo: Repositório de histórico
        """
        self.config = config
        self.trust_repo = trust_repo
        self.learned_weights: Dict[str, float] = {}

    def update_weights(
        self,
        signal_results: List[SignalResult],
        actual_outcome: float
    ):
        """
        Atualiza pesos baseado em outcome real

        Args:
            signal_results: Resultados dos signals
            actual_outcome: Outcome real observado [0, 1]
        """
        # Implementação simplificada
        # Em produção, usar algoritmo de aprendizado (gradient descent, etc.)

        for sr in signal_results:
            # Calcula erro do signal
            error = abs(sr.score - actual_outcome)

            # Ajusta peso (signals com menor erro ganham mais peso)
            current_weight = self.learned_weights.get(sr.signal_name, 1.0)
            adjustment = 0.1 * (1.0 - error)  # Ajuste proporcional ao acerto

            new_weight = current_weight + adjustment
            self.learned_weights[sr.signal_name] = max(0.1, min(2.0, new_weight))

    def get_weights(self, signal_names: List[str]) -> Dict[str, float]:
        """
        Retorna pesos aprendidos

        Args:
            signal_names: Lista de nomes de signals

        Returns:
            Dict {signal_name: weight}
        """
        weights = {}
        for name in signal_names:
            weights[name] = self.learned_weights.get(name, 1.0)

        # Normaliza
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

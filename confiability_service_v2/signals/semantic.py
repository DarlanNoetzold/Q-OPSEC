"""
Semantic Signals - Detecta contradições e drift semântico
"""
from signals.base import TrustSignal
from core.trust_context import TrustContext
from core.trust_result import SignalResult


class SemanticConsistencySignal(TrustSignal):
    """
    Detecta contradições comparando com histórico
    """

    @property
    def name(self) -> str:
        return "semantic_consistency"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """Verifica contradições com dados históricos"""
        try:
            if not context.entity_id:
                return self._create_result(0.7, 0.5, {"reason": "no_entity_id"})

            # Busca histórico da entidade
            history = self.trust_repo.get_entity_history(context.entity_id, limit=5)

            if not history:
                return self._create_result(0.7, 0.5, {"reason": "no_history"})

            # Compara payload atual com histórico (simplificado)
            current_hash = context.payload_hash
            historical_hashes = [event.get("payload_hash", "") for event in history]

            # Se hash idêntico existe, alta consistência
            if current_hash in historical_hashes:
                return self._create_result(1.0, 1.0, {"is_duplicate": True})

            # Caso contrário, score neutro (sem contradição detectada)
            return self._create_result(0.8, 0.7, {"is_consistent": True})

        except Exception as e:
            return self._create_result(0.5, 0.3, {"error": str(e)})


class SemanticDriftSignal(TrustSignal):
    """
    Mede quanto a informação da entidade mudou ao longo do tempo
    """

    @property
    def name(self) -> str:
        return "semantic_drift"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """Calcula drift semântico"""
        try:
            if not context.entity_id:
                return self._create_result(0.7, 0.5, {"reason": "no_entity_id"})

            history = self.trust_repo.get_entity_history(context.entity_id, limit=10)

            if len(history) < 2:
                return self._create_result(0.8, 0.5, {"reason": "insufficient_history"})

            # Conta quantos hashes únicos existem no histórico
            current_hash = context.payload_hash
            historical_hashes = [event.get("payload_hash", "") for event in history]
            unique_hashes = set(historical_hashes + [current_hash])

            # Drift score: quanto mais hashes únicos, maior o drift
            drift_score = len(unique_hashes) / (len(history) + 1)

            # Score inverso (menos drift = melhor)
            score = 1.0 - min(drift_score, 1.0)

            metadata = {
                "drift_score": drift_score,
                "unique_versions": len(unique_hashes),
                "total_events": len(history) + 1,
                "is_drift": drift_score > 0.5
            }

            return self._create_result(score, 0.8, metadata)

        except Exception as e:
            return self._create_result(0.5, 0.3, {"error": str(e)})

"""
Source Signals - Avalia confiabilidade e consistência da fonte
"""
from signals.base import TrustSignal
from core.trust_context import TrustContext
from core.trust_result import SignalResult


class SourceReliabilitySignal(TrustSignal):
    """
    Avalia confiabilidade da fonte baseado em histórico
    """

    @property
    def name(self) -> str:
        return "source_reliability"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """Avalia histórico da fonte"""
        try:
            if not context.source_id:
                return self._create_result(0.5, 0.3, {"reason": "no_source_id"})

            # Busca histórico da fonte
            history = self.trust_repo.get_source_history(context.source_id, limit=20)

            if not history:
                # Fonte nova - score neutro
                return self._create_result(0.5, 0.5, {"reason": "new_source", "history_count": 0})

            # Calcula média de trust scores históricos
            trust_scores = [event.get("trust_score", 0.5) for event in history]
            avg_trust = sum(trust_scores) / len(trust_scores)

            # Confidence aumenta com mais histórico
            confidence = min(1.0, len(history) / 10.0)

            metadata = {
                "avg_trust": avg_trust,
                "history_count": len(history),
                "recent_scores": trust_scores[:5]
            }

            return self._create_result(avg_trust, confidence, metadata)

        except Exception as e:
            return self._create_result(0.5, 0.3, {"error": str(e)})


class SourceConsistencySignal(TrustSignal):
    """
    Verifica consistência estrutural da fonte (payload fingerprints)
    """

    @property
    def name(self) -> str:
        return "source_consistency"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """Verifica se a fonte mantém padrões estruturais consistentes"""
        try:
            if not context.source_id:
                return self._create_result(0.5, 0.3, {"reason": "no_source_id"})

            # Busca fingerprints históricos da fonte
            history = self.trust_repo.get_source_history(context.source_id, limit=10)

            if len(history) < 3:
                # Histórico insuficiente
                return self._create_result(0.7, 0.5, {"reason": "insufficient_history"})

            # Compara fingerprint atual com histórico
            current_fp = context.payload_fp
            historical_fps = [event.get("payload_fp", "") for event in history if event.get("payload_fp")]

            if not historical_fps:
                return self._create_result(0.7, 0.5, {"reason": "no_historical_fingerprints"})

            # Conta quantos fingerprints históricos são similares ao atual
            matches = sum(1 for fp in historical_fps if fp == current_fp)
            consistency_rate = matches / len(historical_fps)

            # Score baseado na taxa de consistência
            score = 0.5 + (consistency_rate * 0.5)  # [0.5, 1.0]

            metadata = {
                "consistency_rate": consistency_rate,
                "matches": matches,
                "total_historical": len(historical_fps),
                "is_consistent": consistency_rate > 0.7
            }

            return self._create_result(score, 0.9, metadata)

        except Exception as e:
            return self._create_result(0.5, 0.3, {"error": str(e)})

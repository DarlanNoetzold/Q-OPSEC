"""
Anomaly Detection Signal - Detecta padrões anômalos
"""
from signals.base import TrustSignal
from core.trust_context import TrustContext
from core.trust_result import SignalResult
from utils.entropy import payload_entropy, is_high_entropy


class AnomalyDetectionSignal(TrustSignal):
    """
    Detecta anomalias usando heurísticas:
    - Entropia alta (dados aleatórios)
    - Tamanho anormal
    - Frequência anormal
    """

    @property
    def name(self) -> str:
        return "anomaly_detection"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """Detecta anomalias no payload"""
        try:
            anomalies = []

            # 1. Verifica entropia
            entropy = payload_entropy(context.payload)
            if is_high_entropy(context.payload, threshold=4.5):
                anomalies.append("high_entropy")

            # 2. Verifica tamanho do payload
            payload_size = len(str(context.payload))
            if payload_size > 10000:
                anomalies.append("large_payload")
            elif payload_size < 10:
                anomalies.append("tiny_payload")

            # 3. Verifica frequência (se entity_id disponível)
            if context.entity_id:
                history = self.trust_repo.get_entity_history(context.entity_id, limit=100)
                if len(history) > 50:
                    anomalies.append("high_frequency")

            # Score baseado no número de anomalias
            score = 1.0 - (len(anomalies) * 0.3)
            score = max(0.0, score)

            metadata = {
                "anomalies": anomalies,
                "entropy": entropy,
                "payload_size": payload_size,
                "is_anomaly": len(anomalies) > 0
            }

            return self._create_result(score, 0.8, metadata)

        except Exception as e:
            return self._create_result(0.5, 0.3, {"error": str(e)})

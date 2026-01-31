"""
Temporal Signals - Avalia freshness e drift temporal
"""
from signals.base import TrustSignal
from core.trust_context import TrustContext
from core.trust_result import SignalResult
from strategies.decay import DecayStrategy
from utils.time import get_age_hours, parse_timestamp
from datetime import datetime


class TemporalSignal(TrustSignal):
    """
    Avalia freshness da informação usando decay temporal
    Informação mais recente = maior confiança
    """

    @property
    def name(self) -> str:
        return "temporal"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """Avalia freshness baseado no timestamp"""
        try:
            # Obtém idade da informação
            age_hours = get_age_hours(context.timestamp)

            # Parâmetros de decay
            decay_lambda = self.params.get("decay_lambda", 0.0001)
            half_life_hours = self.params.get("half_life_hours", 168.0)  # 7 dias

            # Calcula score usando decay exponencial
            score = DecayStrategy.exponential(age_hours, decay_lambda)

            # Confidence baseado na presença de timestamp
            confidence = 1.0 if context.timestamp else 0.5

            metadata = {
                "age_hours": age_hours,
                "decay_lambda": decay_lambda,
                "half_life_hours": half_life_hours,
                "is_stale": age_hours > half_life_hours
            }

            return self._create_result(score, confidence, metadata)

        except Exception as e:
            # Fallback em caso de erro
            return self._create_result(0.5, 0.3, {"error": str(e)})


class TemporalDriftSignal(TrustSignal):
    """
    Detecta inconsistências temporais (timestamps futuros, regressões)
    """

    @property
    def name(self) -> str:
        return "temporal_drift"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """Detecta anomalias temporais"""
        try:
            current_time = datetime.utcnow()
            info_time = context.timestamp

            # Verifica timestamp futuro
            if info_time and info_time > current_time:
                return self._create_result(
                    0.0,
                    1.0,
                    {"is_future": True, "drift_hours": get_age_hours(info_time)}
                )

            # Verifica regressão temporal (comparado com histórico)
            if context.entity_id:
                history = self.trust_repo.get_entity_history(context.entity_id, limit=1)

                if history:
                    last_timestamp = history[0].get("timestamp")
                    if last_timestamp and info_time:
                        last_time = parse_timestamp(last_timestamp)

                        # Se timestamp atual é anterior ao último, há regressão
                        if info_time < last_time:
                            hours_regression = (last_time - info_time).total_seconds() / 3600
                            return self._create_result(
                                0.3,
                                1.0,
                                {"is_regression": True, "regression_hours": hours_regression}
                            )

            # Sem anomalias temporais
            return self._create_result(1.0, 1.0, {"is_drift": False})

        except Exception as e:
            return self._create_result(0.5, 0.3, {"error": str(e)})

"""
Context Signals - Avalia alinhamento e estabilidade de contexto
"""
from signals.base import TrustSignal
from core.trust_context import TrustContext
from core.trust_result import SignalResult


class ContextAlignmentSignal(TrustSignal):
    """
    Verifica se informação está alinhada com contexto esperado
    """

    @property
    def name(self) -> str:
        return "context_alignment"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """Verifica alinhamento contextual"""
        try:
            alignment_score = 1.0
            checks = []

            # 1. Verifica se environment é válido
            valid_envs = ["production", "staging", "development", "test"]
            if context.environment and context.environment not in valid_envs:
                alignment_score -= 0.3
                checks.append(("invalid_environment", False))
            else:
                checks.append(("valid_environment", True))

            # 2. Verifica se data_type é válido
            valid_types = ["event", "profile", "transaction", "log", "metric"]
            if context.data_type and context.data_type not in valid_types:
                alignment_score -= 0.2
                checks.append(("invalid_data_type", False))
            else:
                checks.append(("valid_data_type", True))

            # 3. Verifica completude de metadata
            required_fields = ["source_id", "entity_id"]
            missing = [f for f in required_fields if not getattr(context, f, None)]

            if missing:
                alignment_score -= 0.1 * len(missing)
                checks.append(("missing_metadata", missing))
            else:
                checks.append(("complete_metadata", True))

            alignment_score = max(0.0, alignment_score)

            metadata = {
                "alignment_score": alignment_score,
                "checks": dict(checks),
                "is_aligned": alignment_score > 0.7
            }

            return self._create_result(alignment_score, 0.9, metadata)

        except Exception as e:
            return self._create_result(0.5, 0.3, {"error": str(e)})


class ContextStabilitySignal(TrustSignal):
    """
    Avalia estabilidade do contexto ao longo do tempo
    """

    @property
    def name(self) -> str:
        return "context_stability"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """Verifica se contexto é estável"""
        try:
            if not context.entity_id:
                return self._create_result(0.7, 0.5, {"reason": "no_entity_id"})

            history = self.trust_repo.get_entity_history(context.entity_id, limit=10)

            if len(history) < 3:
                return self._create_result(0.7, 0.5, {"reason": "insufficient_history"})

            # Verifica estabilidade de environment
            envs = [e.get("environment") for e in history if e.get("environment")]
            env_stability = len(set(envs)) == 1 if envs else True

            # Verifica estabilidade de data_type
            types = [e.get("data_type") for e in history if e.get("data_type")]
            type_stability = len(set(types)) == 1 if types else True

            # Score baseado em estabilidade
            stability_score = (int(env_stability) + int(type_stability)) / 2.0

            metadata = {
                "env_stability": env_stability,
                "type_stability": type_stability,
                "unique_envs": len(set(envs)) if envs else 0,
                "unique_types": len(set(types)) if types else 0,
                "is_stable": stability_score > 0.5
            }

            return self._create_result(stability_score, 0.8, metadata)

        except Exception as e:
            return self._create_result(0.5, 0.3, {"error": str(e)})

"""
Consistency Signal - Avalia consistência cross-dimensional
"""
from signals.base import TrustSignal
from core.trust_context import TrustContext
from core.trust_result import SignalResult


class ConsistencySignal(TrustSignal):
    """
    Avalia consistência entre source, entity, data_type, environment
    """

    @property
    def name(self) -> str:
        return "consistency"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """Verifica consistência cross-dimensional"""
        try:
            consistency_checks = []

            # 1. Source-Entity consistency
            if context.source_id and context.entity_id:
                # Verifica se essa combinação já foi vista
                history = self.trust_repo.get_entity_history(context.entity_id, limit=20)
                source_matches = sum(1 for e in history if e.get("source_id") == context.source_id)

                if len(history) > 0:
                    source_consistency = source_matches / len(history)
                    consistency_checks.append(("source_entity", source_consistency))

            # 2. Data type consistency
            if context.entity_id and context.data_type:
                history = self.trust_repo.get_entity_history(context.entity_id, limit=20)
                type_matches = sum(1 for e in history if e.get("data_type") == context.data_type)

                if len(history) > 0:
                    type_consistency = type_matches / len(history)
                    consistency_checks.append(("data_type", type_consistency))

            # 3. Environment consistency
            if context.entity_id and context.environment:
                history = self.trust_repo.get_entity_history(context.entity_id, limit=20)
                env_matches = sum(1 for e in history if e.get("environment") == context.environment)

                if len(history) > 0:
                    env_consistency = env_matches / len(history)
                    consistency_checks.append(("environment", env_consistency))

            if not consistency_checks:
                return self._create_result(0.7, 0.5, {"reason": "no_checks_possible"})

            # Score = média das consistências
            avg_consistency = sum(c[1] for c in consistency_checks) / len(consistency_checks)

            metadata = {
                "consistency_rate": avg_consistency,
                "checks": dict(consistency_checks),
                "is_inconsistent": avg_consistency < 0.5
            }

            return self._create_result(avg_consistency, 0.8, metadata)

        except Exception as e:
            return self._create_result(0.5, 0.3, {"error": str(e)})

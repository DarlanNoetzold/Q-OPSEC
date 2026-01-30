"""
Consistency signal - evaluates cross-dimensional consistency.
"""
from signals.base import TrustSignal
from core.trust_context import TrustContext
from core.trust_result import SignalResult


class ConsistencySignal(TrustSignal):
    """
    Evaluates overall consistency across multiple dimensions.
    """

    @property
    def name(self) -> str:
        return "consistency"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """
        Check consistency across source, entity, and data type.
        """
        if not self.storage:
            return self._create_result(0.5, {"reason": "no_storage"})

        score = 1.0
        flags = []

        # 1. Source-Entity consistency
        source_entity_score = self._check_source_entity_consistency(context)
        score *= source_entity_score
        if source_entity_score < 0.8:
            flags.append("source_entity_mismatch")

        # 2. Data type consistency
        data_type_score = self._check_data_type_consistency(context)
        score *= data_type_score
        if data_type_score < 0.8:
            flags.append("data_type_inconsistency")

        # 3. Environment consistency
        env_score = self._check_environment_consistency(context)
        score *= env_score
        if env_score < 0.8:
            flags.append("environment_mismatch")

        metadata = {
            "flags": flags,
            "source_entity_score": round(source_entity_score, 4),
            "data_type_score": round(data_type_score, 4),
            "environment_score": round(env_score, 4)
        }

        return self._create_result(score, metadata)

    def _check_source_entity_consistency(self, context: TrustContext) -> float:
        """
        Check if source-entity pairing is consistent with history.
        """
        if not context.source_id or not context.entity_id:
            return 1.0

        # Get historical source-entity pairs
        history = self.storage.get_entity_history(
            entity_id=context.entity_id,
            limit=20
        )

        if not history:
            return 1.0

        # Check if this source has interacted with this entity before
        historical_sources = [event.get("source_id") for event in history]

        if context.source_id in historical_sources:
            return 1.0  # consistent

        # New source for this entity
        # Check if entity typically has multiple sources
        unique_sources = len(set(historical_sources))

        if unique_sources > 3:
            return 0.9  # entity has multiple sources, acceptable
        else:
            return 0.7  # entity usually has few sources, suspicious

    def _check_data_type_consistency(self, context: TrustContext) -> float:
        """
        Check if data type is consistent for this entity.
        """
        history = self.storage.get_entity_history(
            entity_id=context.entity_id,
            limit=10
        )

        if not history:
            return 1.0

        historical_types = [event.get("data_type") for event in history]

        if context.data_type in historical_types:
            return 1.0  # consistent

        # New data type for this entity
        unique_types = len(set(historical_types))

        if unique_types > 2:
            return 0.9  # entity has varied data types
        else:
            return 0.75  # unusual new data type

    def _check_environment_consistency(self, context: TrustContext) -> float:
        """
        Check if environment is consistent.
        """
        history = self.storage.get_entity_history(
            entity_id=context.entity_id,
            limit=10
        )

        if not history:
            return 1.0

        historical_envs = [event.get("environment") for event in history]

        if context.environment in historical_envs:
            return 1.0

        # Different environment
        # Prod -> Dev is suspicious, Dev -> Prod is normal
        if "prod" in historical_envs and context.environment != "prod":
            return 0.6  # downgrade from prod is suspicious

        return 0.85  # environment change, but acceptable
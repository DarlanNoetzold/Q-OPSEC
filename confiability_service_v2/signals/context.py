"""
Context alignment signal - evaluates if information fits expected context.
"""
from signals.base import TrustSignal
from core.trust_context import TrustContext
from core.trust_result import SignalResult


class ContextAlignmentSignal(TrustSignal):
    """
    Evaluates if information aligns with expected context.
    """

    @property
    def name(self) -> str:
        return "context_alignment"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """
        Check if context makes sense for this type of information.
        """
        score = 1.0
        flags = []

        # 1. Environment validation
        expected_envs = self.params.get("expected_contexts", ["prod", "staging", "dev"])
        if context.environment not in expected_envs:
            score -= 0.2
            flags.append("unexpected_environment")

        # 2. Source-DataType alignment
        if self._is_misaligned_source_datatype(context):
            score -= 0.15
            flags.append("source_datatype_mismatch")

        # 3. Request ID validation
        if not context.request_id:
            score -= 0.1
            flags.append("missing_request_id")

        # 4. Metadata completeness
        completeness_score = self._check_metadata_completeness(context)
        score *= completeness_score
        if completeness_score < 0.9:
            flags.append("incomplete_metadata")

        metadata = {
            "flags": flags,
            "environment": context.environment,
            "metadata_completeness": round(completeness_score, 4)
        }

        return self._create_result(score, metadata)

    def _is_misaligned_source_datatype(self, context: TrustContext) -> bool:
        """
        Check if source and data type combination makes sense.
        """
        # Example rules (customize based on your domain)
        misalignments = {
            "auth_service": ["payment_data", "analytics_event"],
            "payment_service": ["user_profile", "auth_token"],
            "analytics_service": ["credentials", "payment_data"]
        }

        source_id = context.source_id or ""
        data_type = context.data_type

        for source_pattern, forbidden_types in misalignments.items():
            if source_pattern in source_id and data_type in forbidden_types:
                return True

        return False

    def _check_metadata_completeness(self, context: TrustContext) -> float:
        """
        Check if all expected metadata is present.
        """
        required_fields = ["source_id", "entity_id", "data_type", "timestamp"]
        present_fields = []

        if context.source_id:
            present_fields.append("source_id")
        if context.entity_id:
            present_fields.append("entity_id")
        if context.data_type:
            present_fields.append("data_type")
        if context.timestamp:
            present_fields.append("timestamp")

        completeness = len(present_fields) / len(required_fields)
        return completeness


class ContextStabilitySignal(TrustSignal):
    """
    Evaluates stability of context over time for an entity.
    """

    @property
    def name(self) -> str:
        return "context_stability"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """
        Check if context is stable compared to history.
        """
        if not self.storage or not context.entity_id:
            return self._create_result(0.5, {"reason": "no_history"})

        history = self.storage.get_entity_history(
            entity_id=context.entity_id,
            limit=10
        )

        if len(history) < 3:
            return self._create_result(0.5, {"reason": "insufficient_history"})

        # Check environment stability
        historical_envs = [event.get("environment") for event in history]
        env_changes = len(set(historical_envs))

        # Check source stability
        historical_sources = [event.get("source_id") for event in history]
        source_changes = len(set(historical_sources))

        # Calculate stability score
        # Fewer changes = more stable = higher score
        env_stability = 1.0 - min(env_changes / len(historical_envs), 0.5)
        source_stability = 1.0 - min(source_changes / len(historical_sources), 0.5)

        score = (env_stability + source_stability) / 2

        metadata = {
            "environment_changes": env_changes,
            "source_changes": source_changes,
            "env_stability": round(env_stability, 4),
            "source_stability": round(source_stability, 4)
        }

        return self._create_result(score, metadata)
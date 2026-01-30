"""
Source reliability signal - evaluates trustworthiness based on source history.
"""
from signals.base import TrustSignal
from core.trust_context import TrustContext
from core.trust_result import SignalResult
from typing import Optional


class SourceReliabilitySignal(TrustSignal):
    """
    Evaluates source trustworthiness based on historical behavior.
    Sources build reputation over time.
    """

    @property
    def name(self) -> str:
        return "source_reliability"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """
        Calculate source reliability from historical trust scores.
        """
        source_id = context.source_id

        if not source_id:
            # Unknown source = neutral score
            return self._create_result(0.5, {"reason": "no_source_id"})

        # Get source history from storage
        if not self.storage:
            return self._create_result(0.5, {"reason": "no_storage"})

        history = self.storage.get_source_history(
            source_id=source_id,
            limit=self.params.get("min_history", 10)
        )

        if not history or len(history) < self.params.get("min_history", 3):
            # New source = slightly below neutral (needs to prove itself)
            return self._create_result(
                0.45,
                {"reason": "insufficient_history", "events": len(history)}
            )

        # Calculate average historical trust score
        scores = [event.get("trust_score", 0.5) for event in history]
        avg_score = sum(scores) / len(scores)

        # Apply reputation decay (recent events matter more)
        decay = self.params.get("reputation_decay", 0.95)
        weighted_score = 0.0
        weight_sum = 0.0

        for i, score in enumerate(reversed(scores)):
            weight = decay ** i
            weighted_score += score * weight
            weight_sum += weight

        final_score = weighted_score / weight_sum if weight_sum > 0 else avg_score

        # Detect sudden drops (suspicious)
        if len(scores) >= 3:
            recent_avg = sum(scores[-3:]) / 3
            if recent_avg < avg_score - 0.2:
                final_score *= 0.9  # penalty for degradation

        metadata = {
            "historical_events": len(history),
            "average_score": round(avg_score, 4),
            "weighted_score": round(final_score, 4),
            "source_id": source_id
        }

        return self._create_result(final_score, metadata)


class SourceConsistencySignal(TrustSignal):
    """
    Evaluates consistency of source behavior (structure, patterns).
    """

    @property
    def name(self) -> str:
        return "source_consistency"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """
        Check if source maintains consistent patterns.
        """
        source_id = context.source_id

        if not source_id or not self.storage:
            return self._create_result(0.5, {"reason": "no_data"})

        # Get fingerprint history
        history = self.storage.get_source_fingerprints(
            source_id=source_id,
            limit=10
        )

        if len(history) < 3:
            return self._create_result(0.5, {"reason": "insufficient_data"})

        # Check fingerprint consistency
        current_fp = context.payload_fp
        historical_fps = [event.get("payload_fp") for event in history]

        # Calculate similarity (simple: exact match rate)
        matches = sum(1 for fp in historical_fps if fp == current_fp)
        consistency_rate = matches / len(historical_fps)

        # High consistency = good, but not too rigid
        if consistency_rate > 0.95:
            score = 0.9  # very consistent
        elif consistency_rate > 0.7:
            score = 1.0  # healthy variation
        elif consistency_rate > 0.4:
            score = 0.7  # moderate variation
        else:
            score = 0.5  # high variation (suspicious)

        metadata = {
            "consistency_rate": round(consistency_rate, 4),
            "fingerprint_matches": matches,
            "total_samples": len(historical_fps)
        }

        return self._create_result(score, metadata)
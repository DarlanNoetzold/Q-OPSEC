"""
Temporal trust signal - evaluates information freshness and decay.
"""
from signals.base import TrustSignal
from core.trust_context import TrustContext
from core.trust_result import SignalResult
from strategies.decay import DecayStrategy
from utils.time import now_utc, delta_seconds


class TemporalSignal(TrustSignal):
    """
    Evaluates trust based on temporal validity.
    Older information is less trustworthy.
    """

    @property
    def name(self) -> str:
        return "temporal"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """
        Calculate temporal trust score using exponential decay.

        Score decreases as information ages.
        """
        # Get parameters
        decay_lambda = self.params.get("decay_lambda", 0.0001)
        max_age_hours = self.params.get("max_age_hours", 720)

        # Calculate age
        now = now_utc()
        age_seconds = delta_seconds(now, context.timestamp)
        age_hours = age_seconds / 3600

        # Apply exponential decay
        score = DecayStrategy.exponential(age_seconds, decay_lambda)

        # Hard cutoff at max age
        if age_hours > max_age_hours:
            score = 0.0

        # Metadata
        metadata = {
            "age_seconds": round(age_seconds, 2),
            "age_hours": round(age_hours, 2),
            "decay_applied": round(1.0 - score, 4),
            "is_expired": age_hours > max_age_hours
        }

        return self._create_result(score, metadata)


class TemporalDriftSignal(TrustSignal):
    """
    Detects temporal inconsistencies (e.g., future timestamps, time jumps).
    """

    @property
    def name(self) -> str:
        return "temporal_drift"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """
        Detect temporal anomalies.
        """
        now = now_utc()
        timestamp = context.timestamp

        score = 1.0
        flags = []

        # Future timestamp (suspicious)
        if timestamp > now:
            future_seconds = delta_seconds(timestamp, now)
            if future_seconds > 60:  # allow 1 min clock skew
                score -= 0.3
                flags.append("future_timestamp")

        # Check for time jumps if storage available
        if self.storage:
            last_event = self.storage.get_last_event(
                entity_id=context.entity_id,
                source_id=context.source_id
            )

            if last_event:
                last_ts = last_event.get("timestamp")
                if last_ts and timestamp < last_ts:
                    score -= 0.2
                    flags.append("timestamp_regression")

        metadata = {
            "flags": flags,
            "timestamp": timestamp.isoformat()
        }

        return self._create_result(score, metadata)
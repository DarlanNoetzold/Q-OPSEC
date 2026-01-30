"""
Anomaly detection signal - identifies unusual patterns in information.
"""
from signals.base import TrustSignal
from core.trust_context import TrustContext
from core.trust_result import SignalResult
from utils.entropy import payload_entropy, is_high_entropy
import math
from typing import List


class AnomalyDetectionSignal(TrustSignal):
    """
    Detects anomalies using statistical methods.
    """

    @property
    def name(self) -> str:
        return "anomaly_detection"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """
        Detect anomalies using multiple heuristics.
        """
        score = 1.0
        flags = []

        # 1. Entropy check (detect encoded/encrypted data)
        entropy = payload_entropy(context.payload)
        if is_high_entropy(str(context.payload), threshold=4.5):
            score -= 0.2
            flags.append("high_entropy")

        # 2. Size anomaly
        if self.storage:
            size_score = self._check_size_anomaly(context)
            score *= size_score
            if size_score < 0.8:
                flags.append("size_anomaly")

        # 3. Frequency anomaly
        if self.storage:
            freq_score = self._check_frequency_anomaly(context)
            score *= freq_score
            if freq_score < 0.8:
                flags.append("frequency_anomaly")

        # 4. Structure anomaly
        structure_score = self._check_structure_anomaly(context)
        score *= structure_score
        if structure_score < 0.8:
            flags.append("structure_anomaly")

        metadata = {
            "entropy": round(entropy, 4),
            "flags": flags,
            "checks_performed": 4
        }

        return self._create_result(score, metadata)

    def _check_size_anomaly(self, context: TrustContext) -> float:
        """
        Check if payload size is anomalous compared to history.
        """
        history = self.storage.get_entity_history(
            entity_id=context.entity_id,
            limit=20
        )

        if len(history) < self.params.get("min_samples", 10):
            return 1.0  # not enough data

        # Calculate payload sizes
        current_size = len(str(context.payload))
        historical_sizes = [len(str(event.get("payload", {}))) for event in history]

        # Calculate z-score
        mean_size = sum(historical_sizes) / len(historical_sizes)
        std_size = math.sqrt(
            sum((s - mean_size) ** 2 for s in historical_sizes) / len(historical_sizes)
        )

        if std_size == 0:
            return 1.0

        z_score = abs((current_size - mean_size) / std_size)
        threshold = self.params.get("z_score_threshold", 2.5)

        if z_score > threshold:
            return 0.6  # anomalous size

        return 1.0

    def _check_frequency_anomaly(self, context: TrustContext) -> float:
        """
        Check if events are arriving at unusual frequency.
        """
        history = self.storage.get_entity_history(
            entity_id=context.entity_id,
            limit=10
        )

        if len(history) < 3:
            return 1.0

        # Calculate time deltas between events
        from utils.time import delta_seconds

        timestamps = [event.get("timestamp") for event in history if event.get("timestamp")]

        if len(timestamps) < 2:
            return 1.0

        deltas = []
        for i in range(len(timestamps) - 1):
            from utils.time import parse_iso
            ts1 = parse_iso(timestamps[i]) if isinstance(timestamps[i], str) else timestamps[i]
            ts2 = parse_iso(timestamps[i + 1]) if isinstance(timestamps[i + 1], str) else timestamps[i + 1]
            deltas.append(delta_seconds(ts1, ts2))

        # Check if current event is too fast
        if deltas:
            min_delta = min(deltas)
            avg_delta = sum(deltas) / len(deltas)

            # If events are coming much faster than usual
            if min_delta < avg_delta * 0.1 and avg_delta > 60:
                return 0.7  # suspicious frequency

        return 1.0

    def _check_structure_anomaly(self, context: TrustContext) -> float:
        """
        Check if payload structure is unusual.
        """
        # Check for suspicious patterns
        payload_str = str(context.payload)

        # Too many special characters
        special_chars = sum(1 for c in payload_str if not c.isalnum() and not c.isspace())
        if len(payload_str) > 0 and special_chars / len(payload_str) > 0.5:
            return 0.7

        # Extremely nested structure
        if self._get_depth(context.payload) > 10:
            return 0.8

        return 1.0

    def _get_depth(self, obj, level=0) -> int:
        """Calculate nesting depth."""
        if not isinstance(obj, dict) or not obj:
            return level
        return max(
            self._get_depth(v, level + 1) if isinstance(v, dict) else level + 1
            for v in obj.values()
        )
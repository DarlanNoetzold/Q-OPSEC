"""
Dynamic weighting strategies for trust signals.
"""
from typing import Dict, List
from core.trust_context import TrustContext
from core.trust_result import SignalResult


class DynamicWeighting:
    """
    Adjusts signal weights based on context.
    Different scenarios require different signal priorities.
    """

    def __init__(self, base_weights: Dict[str, float]):
        """
        Initialize with base weights from config.

        Args:
            base_weights: Dictionary of signal_name -> weight
        """
        self.base_weights = base_weights

    def adjust_weights(
            self,
            signals: List[SignalResult],
            context: TrustContext
    ) -> List[SignalResult]:
        """
        Adjust signal weights based on context.

        Args:
            signals: List of signal results
            context: Trust context

        Returns:
            Signals with adjusted weights
        """
        adjusted_signals = []

        for signal in signals:
            adjusted_weight = self._calculate_adjusted_weight(
                signal,
                context
            )

            # Create new signal result with adjusted weight
            adjusted_signal = SignalResult(
                name=signal.name,
                score=signal.score,
                weight=adjusted_weight,
                metadata=signal.metadata
            )
            adjusted_signals.append(adjusted_signal)

        return adjusted_signals

    def _calculate_adjusted_weight(
            self,
            signal: SignalResult,
            context: TrustContext
    ) -> float:
        """Calculate adjusted weight for a signal."""
        base_weight = signal.weight

        # Data type specific adjustments
        if context.data_type == "identity_claim":
            if signal.name == "semantic_consistency":
                base_weight *= 1.3  # identity should be consistent
            elif signal.name == "temporal":
                base_weight *= 0.8  # identity doesn't age as fast

        elif context.data_type == "transaction":
            if signal.name == "temporal":
                base_weight *= 1.5  # transactions age quickly
            elif signal.name == "anomaly_detection":
                base_weight *= 1.4  # anomalies critical for transactions

        elif context.data_type == "analytics_event":
            if signal.name == "temporal":
                base_weight *= 1.2
            elif signal.name == "semantic_consistency":
                base_weight *= 0.7  # analytics can vary more

        # Environment specific adjustments
        if context.environment == "prod":
            if signal.name == "source_reliability":
                base_weight *= 1.2  # source matters more in prod

        elif context.environment == "dev":
            # More lenient in dev
            base_weight *= 0.9

        # New entity adjustments
        if not self._has_history(context):
            if signal.name in ["semantic_consistency", "source_reliability"]:
                base_weight *= 0.5  # can't rely on history
            elif signal.name in ["anomaly_detection", "context_alignment"]:
                base_weight *= 1.3  # rely more on structural checks

        return base_weight

    def _has_history(self, context: TrustContext) -> bool:
        """Check if entity has historical data."""
        # This would check storage, simplified here
        return context.entity_id is not None


class AdaptiveWeighting:
    """
    Learns optimal weights over time (simplified version).
    """

    def __init__(self):
        self.performance_history: Dict[str, List[float]] = {}

    def update_performance(self, signal_name: str, accuracy: float):
        """
        Update performance history for a signal.

        Args:
            signal_name: Name of the signal
            accuracy: How accurate the signal was (0-1)
        """
        if signal_name not in self.performance_history:
            self.performance_history[signal_name] = []

        self.performance_history[signal_name].append(accuracy)

        # Keep only recent history
        if len(self.performance_history[signal_name]) > 100:
            self.performance_history[signal_name] = \
                self.performance_history[signal_name][-100:]

    def get_adaptive_weight(self, signal_name: str, base_weight: float) -> float:
        """
        Get adaptive weight based on historical performance.

        Args:
            signal_name: Name of the signal
            base_weight: Base weight from config

        Returns:
            Adjusted weight
        """
        if signal_name not in self.performance_history:
            return base_weight

        history = self.performance_history[signal_name]

        if len(history) < 10:
            return base_weight  # not enough data

        # Calculate average performance
        avg_performance = sum(history) / len(history)

        # Adjust weight based on performance
        # Good performance (>0.8) -> increase weight
        # Poor performance (<0.5) -> decrease weight
        if avg_performance > 0.8:
            return base_weight * 1.2
        elif avg_performance < 0.5:
            return base_weight * 0.7
        else:
            return base_weight
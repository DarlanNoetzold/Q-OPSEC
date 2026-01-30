"""
Scoring strategies for trust aggregation.
"""
from typing import List, Dict
from core.trust_result import SignalResult
import math


class ScoringStrategy:
    """Different strategies for aggregating signal scores."""

    @staticmethod
    def weighted_average(signals: List[SignalResult]) -> float:
        """
        Simple weighted average.
        score = Σ(signal_score × weight) / Σ(weight)
        """
        if not signals:
            return 0.5

        weighted_sum = sum(s.score * s.weight for s in signals)
        weight_sum = sum(s.weight for s in signals)

        if weight_sum == 0:
            return 0.5

        return weighted_sum / weight_sum

    @staticmethod
    def geometric_mean(signals: List[SignalResult]) -> float:
        """
        Geometric mean with weights.
        More sensitive to low scores (one bad signal hurts more).
        """
        if not signals:
            return 0.5

        # Avoid log(0) by adding small epsilon
        epsilon = 0.001

        weighted_log_sum = sum(
            s.weight * math.log(s.score + epsilon)
            for s in signals
        )
        weight_sum = sum(s.weight for s in signals)

        if weight_sum == 0:
            return 0.5

        result = math.exp(weighted_log_sum / weight_sum) - epsilon
        return max(0.0, min(1.0, result))

    @staticmethod
    def harmonic_mean(signals: List[SignalResult]) -> float:
        """
        Harmonic mean with weights.
        Even more sensitive to low scores than geometric.
        """
        if not signals:
            return 0.5

        epsilon = 0.001

        weighted_reciprocal_sum = sum(
            s.weight / (s.score + epsilon)
            for s in signals
        )
        weight_sum = sum(s.weight for s in signals)

        if weight_sum == 0 or weighted_reciprocal_sum == 0:
            return 0.5

        result = weight_sum / weighted_reciprocal_sum
        return max(0.0, min(1.0, result))

    @staticmethod
    def minimum(signals: List[SignalResult]) -> float:
        """
        Take minimum score (most conservative).
        One bad signal = bad overall score.
        """
        if not signals:
            return 0.5

        return min(s.score for s in signals)

    @staticmethod
    def adaptive(signals: List[SignalResult], context_score: float = 0.5) -> float:
        """
        Adaptive scoring based on signal variance.
        If signals agree -> weighted average
        If signals disagree -> more conservative (geometric mean)
        """
        if not signals:
            return 0.5

        scores = [s.score for s in signals]

        # Calculate variance
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        # High variance = use geometric mean (conservative)
        # Low variance = use weighted average (optimistic)
        if variance > 0.05:  # threshold
            return ScoringStrategy.geometric_mean(signals)
        else:
            return ScoringStrategy.weighted_average(signals)

    @staticmethod
    def confidence_weighted(signals: List[SignalResult]) -> tuple:
        """
        Calculate score with confidence interval.
        Returns (score, confidence_lower, confidence_upper)
        """
        if not signals:
            return (0.5, 0.4, 0.6)

        # Base score
        base_score = ScoringStrategy.weighted_average(signals)

        # Calculate confidence based on signal agreement
        scores = [s.score for s in signals]
        mean_score = sum(scores) / len(scores)
        std_dev = math.sqrt(
            sum((s - mean_score) ** 2 for s in scores) / len(scores)
        )

        # Confidence interval (±1 std dev)
        confidence_lower = max(0.0, base_score - std_dev)
        confidence_upper = min(1.0, base_score + std_dev)

        return (base_score, confidence_lower, confidence_upper)
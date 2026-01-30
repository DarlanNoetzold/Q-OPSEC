"""
Score normalization strategies.
"""
import math
from typing import List


class NormalizationStrategy:
    """Strategies for normalizing trust scores."""

    @staticmethod
    def clamp(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp score to range."""
        return max(min_val, min(max_val, score))

    @staticmethod
    def sigmoid(score: float, center: float = 0.5, steepness: float = 10.0) -> float:
        """
        Apply sigmoid normalization.
        Useful for smoothing extreme values.
        """
        return 1.0 / (1.0 + math.exp(-steepness * (score - center)))

    @staticmethod
    def min_max(score: float, min_score: float, max_score: float) -> float:
        """Min-max normalization to 0-1 range."""
        if max_score == min_score:
            return 0.5
        return (score - min_score) / (max_score - min_score)

    @staticmethod
    def z_score(score: float, mean: float, std: float) -> float:
        """
        Z-score normalization.
        Returns how many standard deviations from mean.
        """
        if std == 0:
            return 0.5
        z = (score - mean) / std
        # Convert to 0-1 range (assuming ±3 std covers most cases)
        return NormalizationStrategy.clamp((z + 3) / 6, 0.0, 1.0)

    @staticmethod
    def softmax(scores: List[float]) -> List[float]:
        """
        Softmax normalization for multiple scores.
        Converts to probability distribution.
        """
        if not scores:
            return []

        exp_scores = [math.exp(s) for s in scores]
        sum_exp = sum(exp_scores)

        if sum_exp == 0:
            return [1.0 / len(scores)] * len(scores)

        return [exp_s / sum_exp for exp_s in exp_scores]

    @staticmethod
    def robust(score: float, percentile_25: float, percentile_75: float) -> float:
        """
        Robust normalization using IQR (Interquartile Range).
        Less sensitive to outliers.
        """
        iqr = percentile_75 - percentile_25
        if iqr == 0:
            return 0.5

        normalized = (score - percentile_25) / iqr
        return NormalizationStrategy.clamp(normalized, 0.0, 1.0)
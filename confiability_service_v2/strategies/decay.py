"""
Temporal decay strategies for trust scoring.
"""
import math
from datetime import datetime
from typing import Callable
from utils.time import delta_seconds


class DecayStrategy:
    """Base class for decay strategies."""

    @staticmethod
    def exponential(delta_seconds: float, lambda_param: float = 0.0001) -> float:
        """
        Exponential decay: score = e^(-λ * Δt)

        Args:
            delta_seconds: Time difference in seconds
            lambda_param: Decay rate (higher = faster decay)

        Returns:
            Decay multiplier (0.0 to 1.0)
        """
        return math.exp(-lambda_param * delta_seconds)

    @staticmethod
    def linear(delta_seconds: float, max_age_seconds: float) -> float:
        """
        Linear decay: score = 1 - (Δt / max_age)

        Args:
            delta_seconds: Time difference in seconds
            max_age_seconds: Maximum age before score reaches 0

        Returns:
            Decay multiplier (0.0 to 1.0)
        """
        if delta_seconds >= max_age_seconds:
            return 0.0
        return 1.0 - (delta_seconds / max_age_seconds)

    @staticmethod
    def logarithmic(delta_seconds: float, scale: float = 3600) -> float:
        """
        Logarithmic decay: score = 1 / (1 + log(1 + Δt/scale))
        Slower decay initially, faster later.

        Args:
            delta_seconds: Time difference in seconds
            scale: Time scale factor (default: 1 hour)

        Returns:
            Decay multiplier (0.0 to 1.0)
        """
        return 1.0 / (1.0 + math.log(1.0 + delta_seconds / scale))

    @staticmethod
    def step(delta_seconds: float, thresholds: list) -> float:
        """
        Step decay: discrete levels based on age.

        Args:
            delta_seconds: Time difference in seconds
            thresholds: List of (age_seconds, score) tuples

        Returns:
            Decay multiplier (0.0 to 1.0)
        """
        for age, score in sorted(thresholds, reverse=True):
            if delta_seconds >= age:
                return score
        return 1.0

    @staticmethod
    def adaptive(
            delta_seconds: float,
            data_type: str,
            type_configs: dict
    ) -> float:
        """
        Adaptive decay based on data type.
        Different data types age differently.

        Args:
            delta_seconds: Time difference in seconds
            data_type: Type of data being evaluated
            type_configs: Configuration mapping data_type -> decay params

        Returns:
            Decay multiplier (0.0 to 1.0)
        """
        config = type_configs.get(data_type, {"lambda": 0.0001})
        lambda_param = config.get("lambda", 0.0001)
        return DecayStrategy.exponential(delta_seconds, lambda_param)


def get_decay_function(strategy: str = "exponential") -> Callable:
    """
    Get decay function by name.

    Args:
        strategy: Name of decay strategy

    Returns:
        Decay function
    """
    strategies = {
        "exponential": DecayStrategy.exponential,
        "linear": DecayStrategy.linear,
        "logarithmic": DecayStrategy.logarithmic,
        "step": DecayStrategy.step,
        "adaptive": DecayStrategy.adaptive
    }
    return strategies.get(strategy, DecayStrategy.exponential)
"""
Base interface for trust signals.
All signals must inherit from TrustSignal.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional
from core.trust_context import TrustContext
from core.trust_result import SignalResult
from config.trust_config import SignalConfig


class TrustSignal(ABC):
    """
    Abstract base class for trust signals.
    Each signal evaluates one dimension of trust.
    """

    def __init__(self, config: SignalConfig, storage=None):
        """
        Initialize signal with configuration.

        Args:
            config: Signal-specific configuration
            storage: Optional storage backend for historical data
        """
        self.config = config
        self.storage = storage
        self.enabled = config.enabled
        self.weight = config.weight
        self.params = config.params

    @property
    @abstractmethod
    def name(self) -> str:
        """Signal identifier."""
        pass

    @abstractmethod
    def evaluate(self, context: TrustContext) -> SignalResult:
        """
        Evaluate trust signal for given context.

        Args:
            context: Trust evaluation context

        Returns:
            SignalResult with score (0.0 to 1.0) and metadata
        """
        pass

    def _create_result(
            self,
            score: float,
            metadata: Optional[Dict] = None
    ) -> SignalResult:
        """
        Helper to create SignalResult with proper normalization.

        Args:
            score: Raw score (will be clamped to 0.0-1.0)
            metadata: Optional metadata dictionary
        """
        normalized_score = max(0.0, min(1.0, score))
        return SignalResult(
            name=self.name,
            score=normalized_score,
            weight=self.weight,
            metadata=metadata or {}
        )

    def is_enabled(self) -> bool:
        """Check if signal is enabled."""
        return self.enabled

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight={self.weight}, enabled={self.enabled})"
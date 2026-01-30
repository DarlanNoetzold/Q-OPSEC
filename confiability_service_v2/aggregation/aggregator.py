"""
Trust aggregator - combines signal results into final trust score.
"""
from typing import List, Dict
from core.trust_context import TrustContext
from core.trust_result import SignalResult, TrustResult
from aggregation.weighting import DynamicWeighting
from aggregation.explainability import ExplainabilityGenerator
from strategies.scoring import ScoringStrategy
from config.trust_config import TrustConfig


class TrustAggregator:
    """
    Aggregates multiple trust signals into a final trust score.
    """

    def __init__(self, config: TrustConfig):
        """
        Initialize aggregator with configuration.

        Args:
            config: Trust engine configuration
        """
        self.config = config
        self.weighting = DynamicWeighting(
            {name: cfg.weight for name, cfg in config.signals.items()}
        )
        self.explainability = ExplainabilityGenerator()

    def aggregate(
            self,
            signals: List[SignalResult],
            context: TrustContext
    ) -> TrustResult:
        """
        Aggregate signal results into final trust result.

        Args:
            signals: List of evaluated signal results
            context: Trust evaluation context

        Returns:
            Complete trust result with score, dimensions, and explainability
        """
        if not signals:
            return self._create_default_result()

        # 1. Apply dynamic weighting
        adjusted_signals = self.weighting.adjust_weights(signals, context)

        # 2. Calculate final score with confidence interval
        final_score, conf_lower, conf_upper = ScoringStrategy.confidence_weighted(
            adjusted_signals
        )

        # 3. Determine trust level
        trust_level = self.config.get_trust_level(final_score)

        # 4. Extract dimensional scores
        dimensions = self._extract_dimensions(adjusted_signals)

        # 5. Identify risk flags
        risk_flags = self._identify_risk_flags(adjusted_signals, final_score)

        # 6. Generate explainability
        explainability = self.explainability.generate(adjusted_signals)

        # 7. Create result
        result = TrustResult(
            trust_score=final_score,
            trust_level=trust_level,
            confidence_interval=(conf_lower, conf_upper),
            dimensions=dimensions,
            risk_flags=risk_flags,
            explainability=explainability,
            metadata={
                "signals_evaluated": len(signals),
                "context_type": context.data_type,
                "environment": context.environment
            }
        )

        return result

    def _extract_dimensions(self, signals: List[SignalResult]) -> Dict[str, float]:
        """
        Extract dimensional scores from signals.
        Maps signal names to dimension names.
        """
        dimension_mapping = {
            "source_reliability": "source_reliability",
            "source_consistency": "source_reliability",
            "semantic_consistency": "semantic_consistency",
            "semantic_drift": "semantic_consistency",
            "temporal": "temporal_validity",
            "temporal_drift": "temporal_validity",
            "context_alignment": "context_alignment",
            "context_stability": "context_alignment",
            "anomaly_detection": "anomaly_score",
            "consistency": "consistency_score"
        }

        dimensions = {}
        dimension_counts = {}

        for signal in signals:
            dimension = dimension_mapping.get(signal.name, signal.name)

            if dimension not in dimensions:
                dimensions[dimension] = 0.0
                dimension_counts[dimension] = 0

            dimensions[dimension] += signal.score
            dimension_counts[dimension] += 1

        # Average scores for each dimension
        for dimension in dimensions:
            if dimension_counts[dimension] > 0:
                dimensions[dimension] /= dimension_counts[dimension]

        return dimensions

    def _identify_risk_flags(
            self,
            signals: List[SignalResult],
            final_score: float
    ) -> List[str]:
        """
        Identify risk flags from signal results.
        """
        flags = []

        # Check for critically low signals
        for signal in signals:
            if signal.score < 0.3:
                flags.append(f"critical_{signal.name}")
            elif signal.score < 0.5:
                flags.append(f"low_{signal.name}")

        # Check for high variance (signals disagree)
        scores = [s.score for s in signals]
        if len(scores) > 1:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

            if variance > 0.1:
                flags.append("high_signal_variance")

        # Check metadata flags from signals
        for signal in signals:
            metadata_flags = signal.metadata.get("flags", [])
            flags.extend(metadata_flags)

        # Overall score flags
        if final_score < 0.3:
            flags.append("critical_trust_score")
        elif final_score < 0.5:
            flags.append("low_trust_score")

        # Remove duplicates and return
        return list(set(flags))

    def _create_default_result(self) -> TrustResult:
        """Create default result when no signals available."""
        return TrustResult(
            trust_score=0.5,
            trust_level="MEDIUM",
            confidence_interval=(0.4, 0.6),
            dimensions={},
            risk_flags=["no_signals_evaluated"],
            explainability=[],
            metadata={"reason": "no_signals"}
        )
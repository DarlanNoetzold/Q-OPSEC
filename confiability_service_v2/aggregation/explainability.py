"""
Explainability generation for trust scores.
"""
from typing import List, Dict
from core.trust_result import SignalResult


class ExplainabilityGenerator:
    """
    Generates human-readable explanations for trust scores.
    """

    def generate(self, signals: List[SignalResult]) -> List[Dict]:
        """
        Generate explainability entries from signal results.

        Args:
            signals: List of evaluated signals

        Returns:
            List of explainability dictionaries
        """
        explanations = []

        for signal in signals:
            explanation = self._generate_signal_explanation(signal)
            if explanation:
                explanations.append(explanation)

        # Sort by absolute impact (most influential first)
        explanations.sort(key=lambda x: abs(x["impact"]), reverse=True)

        return explanations

    def _generate_signal_explanation(self, signal: SignalResult) -> Dict:
        """Generate explanation for a single signal."""
        impact = signal.impact
        reason = self._generate_reason(signal)

        return {
            "signal": signal.name,
            "impact": round(impact, 4),
            "score": round(signal.score, 4),
            "reason": reason
        }

    def _generate_reason(self, signal: SignalResult) -> str:
        """
        Generate human-readable reason for signal score.
        """
        metadata = signal.metadata
        score = signal.score

        # Temporal signal
        if signal.name == "temporal":
            age_hours = metadata.get("age_hours", 0)
            if score > 0.9:
                return f"Information is fresh ({age_hours:.1f}h old)"
            elif score > 0.7:
                return f"Information is recent ({age_hours:.1f}h old)"
            elif score > 0.5:
                return f"Information is aging ({age_hours:.1f}h old)"
            else:
                return f"Information is stale ({age_hours:.1f}h old)"

        # Source reliability
        elif signal.name == "source_reliability":
            events = metadata.get("historical_events", 0)
            if score > 0.8:
                return f"Source has strong track record ({events} events)"
            elif score > 0.6:
                return f"Source has moderate history ({events} events)"
            elif score > 0.4:
                return f"Source is relatively new ({events} events)"
            else:
                return f"Source has poor history ({events} events)"

        # Semantic consistency
        elif signal.name == "semantic_consistency":
            contradictions = metadata.get("contradictions", 0)
            if contradictions > 0:
                return f"Found {contradictions} contradiction(s) with history"
            elif score > 0.8:
                return "Information is consistent with history"
            else:
                return "Limited historical data for comparison"

        # Anomaly detection
        elif signal.name == "anomaly_detection":
            flags = metadata.get("flags", [])
            if flags:
                return f"Detected anomalies: {', '.join(flags)}"
            else:
                return "No anomalies detected"

        # Consistency
        elif signal.name == "consistency":
            flags = metadata.get("flags", [])
            if flags:
                return f"Consistency issues: {', '.join(flags)}"
            else:
                return "Cross-dimensional consistency verified"

        # Context alignment
        elif signal.name == "context_alignment":
            flags = metadata.get("flags", [])
            if flags:
                return f"Context issues: {', '.join(flags)}"
            else:
                return "Context aligns with expectations"

        # Generic fallback
        if score > 0.8:
            return f"Signal indicates high trust"
        elif score > 0.6:
            return f"Signal indicates moderate trust"
        elif score > 0.4:
            return f"Signal indicates low trust"
        else:
            return f"Signal indicates critical trust issues"

    def generate_summary(self, signals: List[SignalResult], final_score: float) -> str:
        """
        Generate overall summary of trust evaluation.

        Args:
            signals: List of evaluated signals
            final_score: Final aggregated trust score

        Returns:
            Human-readable summary
        """
        # Find most impactful signals
        sorted_signals = sorted(signals, key=lambda s: abs(s.impact), reverse=True)
        top_positive = [s for s in sorted_signals if s.impact > 0][:2]
        top_negative = [s for s in sorted_signals if s.impact < 0][:2]

        summary_parts = []

        if final_score > 0.8:
            summary_parts.append("Information is highly trustworthy.")
        elif final_score > 0.6:
            summary_parts.append("Information is moderately trustworthy.")
        elif final_score > 0.4:
            summary_parts.append("Information has low trustworthiness.")
        else:
            summary_parts.append("Information has critical trust issues.")

        if top_positive:
            positive_names = [s.name.replace("_", " ") for s in top_positive]
            summary_parts.append(
                f"Positive factors: {', '.join(positive_names)}."
            )

        if top_negative:
            negative_names = [s.name.replace("_", " ") for s in top_negative]
            summary_parts.append(
                f"Concerns: {', '.join(negative_names)}."
            )

        return " ".join(summary_parts)
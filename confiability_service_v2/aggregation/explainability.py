"""
Explainability Generator - Gera explicações human-readable para trust scores
"""
from typing import List, Dict, Any
from core.trust_result import SignalResult
from config.trust_config import TrustConfig


class ExplainabilityGenerator:
    """
    Gera explicações detalhadas e human-readable para:
    - Scores individuais de signals
    - Trust score final
    - Risk flags
    - Recomendações
    """

    def __init__(self, config: TrustConfig):
        """
        Inicializa o gerador de explicações

        Args:
            config: Configuração do Trust Engine
        """
        self.config = config

    def generate_explanation(
        self,
        signal_results: List[SignalResult],
        weights: Dict[str, float],
        trust_score: float,
        trust_level: str
    ) -> Dict[str, Any]:
        """
        Gera explicação completa da avaliação

        Args:
            signal_results: Resultados dos signals
            weights: Pesos aplicados
            trust_score: Score final
            trust_level: Nível de confiança

        Returns:
            Dict com summary, details, recommendations
        """
        # 1. Summary geral
        summary = self._generate_summary(trust_score, trust_level, signal_results)

        # 2. Detalhes por signal
        details = self._generate_signal_details(signal_results, weights)

        # 3. Recomendações
        recommendations = self._generate_recommendations(
            trust_score, trust_level, signal_results
        )

        # 4. Fatores principais (top contributors)
        top_factors = self._identify_top_factors(signal_results, weights)

        return {
            "summary": summary,
            "details": details,
            "recommendations": recommendations,
            "top_factors": top_factors
        }

    def _generate_summary(
        self,
        trust_score: float,
        trust_level: str,
        signal_results: List[SignalResult]
    ) -> str:
        """Gera summary geral da avaliação"""
        num_signals = len(signal_results)
        avg_confidence = sum(sr.confidence for sr in signal_results) / num_signals if num_signals > 0 else 0

        # Identifica signals problemáticos
        low_signals = [sr for sr in signal_results if sr.score < 0.4]

        summary = f"Trust evaluation completed with {trust_level} trust level (score: {trust_score:.2f}). "
        summary += f"Analyzed {num_signals} signals with average confidence of {avg_confidence:.2f}. "

        if low_signals:
            summary += f"⚠️ {len(low_signals)} signal(s) raised concerns: "
            summary += ", ".join([sr.signal_name for sr in low_signals[:3]])
            if len(low_signals) > 3:
                summary += f" and {len(low_signals) - 3} more."
        else:
            summary += "✅ All signals passed validation."

        return summary

    def _generate_signal_details(
        self,
        signal_results: List[SignalResult],
        weights: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Gera detalhes para cada signal"""
        details = []

        for sr in signal_results:
            weight = weights.get(sr.signal_name, 0.0)

            detail = {
                "signal": sr.signal_name,
                "score": round(sr.score, 3),
                "confidence": round(sr.confidence, 3),
                "weight": round(weight, 3),
                "contribution": round(sr.score * weight, 3),
                "explanation": self._explain_signal(sr),
                "metadata": sr.metadata
            }

            details.append(detail)

        # Ordena por contribuição (maior primeiro)
        details.sort(key=lambda d: d["contribution"], reverse=True)

        return details

    def _explain_signal(self, signal_result: SignalResult) -> str:
        """Gera explicação human-readable para um signal específico"""
        signal_name = signal_result.signal_name
        score = signal_result.score
        metadata = signal_result.metadata

        # Explicações específicas por tipo de signal
        if "temporal" in signal_name.lower():
            if "age_hours" in metadata:
                age = metadata["age_hours"]
                if score > 0.8:
                    return f"Information is fresh ({age:.1f}h old)"
                elif score > 0.5:
                    return f"Information is moderately fresh ({age:.1f}h old)"
                else:
                    return f"⚠️ Information is stale ({age:.1f}h old)"

        elif "source" in signal_name.lower():
            if "avg_trust" in metadata:
                avg_trust = metadata["avg_trust"]
                if score > 0.8:
                    return f"Source has excellent track record (avg: {avg_trust:.2f})"
                elif score > 0.5:
                    return f"Source has acceptable track record (avg: {avg_trust:.2f})"
                else:
                    return f"⚠️ Source has poor track record (avg: {avg_trust:.2f})"

        elif "semantic" in signal_name.lower():
            if "drift_score" in metadata:
                drift = metadata["drift_score"]
                if score > 0.8:
                    return f"Information is consistent with history (drift: {drift:.2f})"
                else:
                    return f"⚠️ Information shows semantic drift (drift: {drift:.2f})"

        elif "anomaly" in signal_name.lower():
            if "is_anomaly" in metadata and metadata["is_anomaly"]:
                return f"⚠️ Anomalous patterns detected"
            else:
                return "No anomalies detected"

        elif "consistency" in signal_name.lower():
            if "consistency_rate" in metadata:
                rate = metadata["consistency_rate"]
                if score > 0.8:
                    return f"Highly consistent with historical patterns ({rate:.1%})"
                else:
                    return f"⚠️ Inconsistent with historical patterns ({rate:.1%})"

        elif "context" in signal_name.lower():
            if "alignment_score" in metadata:
                alignment = metadata["alignment_score"]
                if score > 0.8:
                    return f"Well-aligned with expected context ({alignment:.2f})"
                else:
                    return f"⚠️ Misaligned with expected context ({alignment:.2f})"

        # Explicação genérica
        if score > 0.8:
            return f"Signal passed with high confidence"
        elif score > 0.5:
            return f"Signal passed with moderate confidence"
        else:
            return f"⚠️ Signal raised concerns"

    def _generate_recommendations(
        self,
        trust_score: float,
        trust_level: str,
        signal_results: List[SignalResult]
    ) -> List[str]:
        """Gera recomendações baseadas na avaliação"""
        recommendations = []

        # Recomendações por trust level
        if trust_level == "VERY_LOW":
            recommendations.append("🚨 REJECT this information - trust score is critically low")
            recommendations.append("Investigate source reliability and data integrity")
        elif trust_level == "LOW":
            recommendations.append("⚠️ Use with EXTREME caution - trust score is low")
            recommendations.append("Require additional validation before use")
        elif trust_level == "MEDIUM":
            recommendations.append("⚠️ Use with caution - trust score is moderate")
            recommendations.append("Consider cross-referencing with other sources")
        elif trust_level == "HIGH":
            recommendations.append("✅ Information appears trustworthy")
            recommendations.append("Standard validation procedures apply")
        else:  # VERY_HIGH
            recommendations.append("✅ Information is highly trustworthy")
            recommendations.append("Minimal additional validation required")

        # Recomendações específicas por signal
        for sr in signal_results:
            if sr.score < 0.3:
                if "temporal" in sr.signal_name.lower():
                    recommendations.append("📅 Update information - data is stale")
                elif "source" in sr.signal_name.lower():
                    recommendations.append("🔍 Verify source reliability")
                elif "semantic" in sr.signal_name.lower():
                    recommendations.append("📝 Check for contradictions or drift")
                elif "anomaly" in sr.signal_name.lower():
                    recommendations.append("🚨 Investigate anomalous patterns")

        return list(set(recommendations))  # Remove duplicatas

    def _identify_top_factors(
        self,
        signal_results: List[SignalResult],
        weights: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Identifica os top 3 fatores que mais influenciaram o score"""
        factors = []

        for sr in signal_results:
            weight = weights.get(sr.signal_name, 0.0)
            contribution = sr.score * weight

            factors.append({
                "signal": sr.signal_name,
                "contribution": round(contribution, 3),
                "impact": "positive" if sr.score > 0.7 else "negative" if sr.score < 0.4 else "neutral"
            })

        # Ordena por contribuição absoluta (maior impacto)
        factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)

        return factors[:3]  # Top 3

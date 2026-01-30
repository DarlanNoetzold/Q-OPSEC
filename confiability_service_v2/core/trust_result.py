"""
Trust evaluation result - encapsulates the output of trust assessment.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from utils.hashing import trust_dna


@dataclass
class SignalResult:
    """Result from a single trust signal."""
    name: str
    score: float  # 0.0 to 1.0
    weight: float
    impact: float = field(init=False)  # weighted contribution
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.impact = (self.score - 0.5) * self.weight  # centered impact


@dataclass
class TrustResult:
    """
    Complete trust evaluation result.
    Includes score, dimensions, explainability, and trust DNA.
    """

    # Core score
    trust_score: float  # 0.0 to 1.0
    trust_level: str  # CRITICAL, LOW, MEDIUM, HIGH, VERIFIED

    # Confidence
    confidence_interval: Tuple[float, float]

    # Dimensional breakdown
    dimensions: Dict[str, float]

    # Risk indicators
    risk_flags: List[str] = field(default_factory=list)

    # Explainability
    explainability: List[Dict[str, any]] = field(default_factory=list)

    # Trust DNA
    trust_dna_value: str = field(init=False)

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Generate trust DNA from dimensions."""
        vector = [
            self.dimensions.get("source_reliability", 0.5),
            self.dimensions.get("semantic_consistency", 0.5),
            self.dimensions.get("temporal_validity", 0.5),
            self.dimensions.get("context_alignment", 0.5),
            self.dimensions.get("anomaly_score", 0.5),
            self.dimensions.get("consistency_score", 0.5)
        ]
        self.trust_dna_value = trust_dna(vector)

    def to_dict(self) -> Dict:
        """Convert to API response format."""
        return {
            "trust_score": round(self.trust_score, 4),
            "trust_level": self.trust_level,
            "confidence_interval": [
                round(self.confidence_interval[0], 4),
                round(self.confidence_interval[1], 4)
            ],
            "dimensions": {k: round(v, 4) for k, v in self.dimensions.items()},
            "risk_flags": self.risk_flags,
            "trust_dna": self.trust_dna_value,
            "explainability": self.explainability,
            "metadata": self.metadata
        }

    def add_risk_flag(self, flag: str):
        """Add a risk flag."""
        if flag not in self.risk_flags:
            self.risk_flags.append(flag)

    def add_explanation(self, signal: str, impact: float, reason: str = ""):
        """Add an explainability entry."""
        self.explainability.append({
            "signal": signal,
            "impact": round(impact, 4),
            "reason": reason
        })

    @property
    def is_trustworthy(self) -> bool:
        """Quick check if result is trustworthy (>= MEDIUM)."""
        return self.trust_score >= 0.5

    @property
    def requires_review(self) -> bool:
        """Check if result requires human review."""
        return self.trust_level in ["CRITICAL", "LOW"] or len(self.risk_flags) > 2
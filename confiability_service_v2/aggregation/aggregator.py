"""
Trust Aggregator - Orquestra a agregação de signals em um TrustResult final
"""
from typing import List, Dict, Any
from core.trust_context import TrustContext
from core.trust_result import SignalResult, TrustResult
from config.trust_config import TrustConfig
from storage.trust_repository import TrustRepository
from aggregation.weighting import DynamicWeighting
from aggregation.explainability import ExplainabilityGenerator
from strategies.scoring import ScoringStrategy
from strategies.normalization import NormalizationStrategy
from utils.hashing import trust_dna


class TrustAggregator:
    """
    Agrega múltiplos SignalResults em um TrustResult final
    Aplica pesos dinâmicos, normalização e gera explainability
    """

    def __init__(
        self,
        config: TrustConfig,
        trust_repo: TrustRepository,
        weighting_strategy: str = "dynamic",
        scoring_strategy: str = "confidence_weighted"
    ):
        """
        Inicializa o agregador

        Args:
            config: Configuração do Trust Engine
            trust_repo: Repositório de histórico
            weighting_strategy: Estratégia de pesos ("dynamic", "adaptive", "static")
            scoring_strategy: Estratégia de scoring ("weighted_average", "confidence_weighted", etc)
        """
        self.config = config
        self.trust_repo = trust_repo
        self.weighting_strategy = weighting_strategy
        self.scoring_strategy = scoring_strategy

        # Inicializa componentes
        self.dynamic_weighting = DynamicWeighting(config, trust_repo)
        self.explainability = ExplainabilityGenerator(config)

    def aggregate(
        self,
        context: TrustContext,
        signal_results: List[SignalResult]
    ) -> TrustResult:
        """
        Agrega os resultados dos signals em um TrustResult final

        Args:
            context: Contexto da avaliação
            signal_results: Lista de resultados dos signals

        Returns:
            TrustResult com score, level, dimensions, flags, explainability
        """
        if not signal_results:
            return self._create_empty_result(context)

        # 1. Calcula pesos dinâmicos
        weights = self._calculate_weights(context, signal_results)

        # 2. Calcula trust score e confidence interval
        trust_score, confidence_interval = self._calculate_trust_score(
            signal_results, weights
        )

        # 3. Determina trust level
        trust_level = self.config.get_trust_level(trust_score)

        # 4. Extrai dimensões
        dimensions = self._extract_dimensions(signal_results, weights)

        # 5. Identifica risk flags
        risk_flags = self._identify_risk_flags(signal_results, trust_score)

        # 6. Gera explainability
        explainability = self.explainability.generate_explanation(
            signal_results, weights, trust_score, trust_level
        )

        # 7. Calcula Trust DNA
        trust_dna_value = self._calculate_trust_dna(signal_results, weights)

        return TrustResult(
            trust_score=trust_score,
            trust_level=trust_level,
            confidence_interval=confidence_interval,
            dimensions=dimensions,
            risk_flags=risk_flags,
            explainability=explainability,
            trust_dna_value=trust_dna_value,
            context=context,
            signal_results=signal_results
        )

    def _calculate_weights(
        self,
        context: TrustContext,
        signal_results: List[SignalResult]
    ) -> Dict[str, float]:
        """Calcula pesos dinâmicos para cada signal"""
        if self.weighting_strategy == "dynamic":
            return self.dynamic_weighting.calculate_weights(context, signal_results)
        elif self.weighting_strategy == "static":
            # Pesos iguais
            return {sr.signal_name: 1.0 / len(signal_results) for sr in signal_results}
        else:
            # Default: dynamic
            return self.dynamic_weighting.calculate_weights(context, signal_results)

    def _calculate_trust_score(
        self,
        signal_results: List[SignalResult],
        weights: Dict[str, float]
    ) -> tuple[float, tuple[float, float]]:
        """
        Calcula trust score e confidence interval

        Returns:
            (trust_score, (lower_bound, upper_bound))
        """
        if self.scoring_strategy == "confidence_weighted":
            return ScoringStrategy.confidence_weighted(signal_results, weights)
        elif self.scoring_strategy == "weighted_average":
            score = ScoringStrategy.weighted_average(signal_results, weights)
            return score, (score * 0.95, min(1.0, score * 1.05))
        elif self.scoring_strategy == "geometric_mean":
            score = ScoringStrategy.geometric_mean(signal_results, weights)
            return score, (score * 0.95, min(1.0, score * 1.05))
        else:
            # Default: confidence_weighted
            return ScoringStrategy.confidence_weighted(signal_results, weights)

    def _extract_dimensions(
        self,
        signal_results: List[SignalResult],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Extrai dimensões de confiança (temporal, source, semantic, etc)
        Agrupa signals por categoria
        """
        dimensions = {
            "temporal": 0.0,
            "source": 0.0,
            "semantic": 0.0,
            "anomaly": 0.0,
            "consistency": 0.0,
            "context": 0.0
        }

        dimension_counts = {k: 0 for k in dimensions.keys()}

        for sr in signal_results:
            # Mapeia signal para dimensão
            dimension = self._map_signal_to_dimension(sr.signal_name)
            if dimension in dimensions:
                weight = weights.get(sr.signal_name, 0.0)
                dimensions[dimension] += sr.score * weight
                dimension_counts[dimension] += 1

        # Normaliza (média ponderada por dimensão)
        for dim in dimensions:
            if dimension_counts[dim] > 0:
                dimensions[dim] = dimensions[dim] / dimension_counts[dim]

        return dimensions

    def _map_signal_to_dimension(self, signal_name: str) -> str:
        """Mapeia nome do signal para dimensão"""
        if "temporal" in signal_name.lower():
            return "temporal"
        elif "source" in signal_name.lower():
            return "source"
        elif "semantic" in signal_name.lower():
            return "semantic"
        elif "anomaly" in signal_name.lower():
            return "anomaly"
        elif "consistency" in signal_name.lower():
            return "consistency"
        elif "context" in signal_name.lower():
            return "context"
        else:
            return "consistency"  # Default

    def _identify_risk_flags(
        self,
        signal_results: List[SignalResult],
        trust_score: float
    ) -> List[str]:
        """Identifica risk flags baseado nos signals"""
        flags = []

        # Flag: Low trust score
        if trust_score < self.config.trust_thresholds["low"]:
            flags.append("LOW_TRUST_SCORE")

        # Flag: Signals com score muito baixo
        for sr in signal_results:
            if sr.score < 0.3:
                flags.append(f"LOW_{sr.signal_name.upper()}_SCORE")

        # Flag: Alta variância entre signals
        scores = [sr.score for sr in signal_results]
        if len(scores) > 1:
            variance = sum((s - trust_score) ** 2 for s in scores) / len(scores)
            if variance > 0.1:
                flags.append("HIGH_SIGNAL_VARIANCE")

        # Flag: Baixa confidence em múltiplos signals
        low_confidence_count = sum(1 for sr in signal_results if sr.confidence < 0.5)
        if low_confidence_count > len(signal_results) / 2:
            flags.append("LOW_CONFIDENCE")

        # Flags específicos de metadata dos signals
        for sr in signal_results:
            if sr.metadata.get("is_anomaly"):
                flags.append("ANOMALY_DETECTED")
            if sr.metadata.get("is_drift"):
                flags.append("SEMANTIC_DRIFT")
            if sr.metadata.get("is_stale"):
                flags.append("STALE_INFORMATION")
            if sr.metadata.get("is_inconsistent"):
                flags.append("INCONSISTENT_DATA")

        return list(set(flags))  # Remove duplicatas

    def _calculate_trust_dna(
        self,
        signal_results: List[SignalResult],
        weights: Dict[str, float]
    ) -> str:
        """
        Calcula Trust DNA (fingerprint único da avaliação)
        Baseado nos scores ponderados dos signals
        """
        # Ordena signals por nome para consistência
        sorted_results = sorted(signal_results, key=lambda sr: sr.signal_name)

        # Cria vetor de scores ponderados
        weighted_scores = [
            sr.score * weights.get(sr.signal_name, 0.0)
            for sr in sorted_results
        ]

        return trust_dna(weighted_scores)

    def _create_empty_result(self, context: TrustContext) -> TrustResult:
        """Cria resultado vazio quando não há signals"""
        return TrustResult(
            trust_score=0.5,
            trust_level="MEDIUM",
            confidence_interval=(0.0, 1.0),
            dimensions={},
            risk_flags=["NO_SIGNALS"],
            explainability={
                "summary": "No signals available for evaluation",
                "details": []
            },
            trust_dna_value="",
            context=context,
            signal_results=[]
        )

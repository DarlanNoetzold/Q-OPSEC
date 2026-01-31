"""
Scoring Strategies - Diferentes estratégias para agregar scores de signals
"""
from typing import List, Dict, Tuple
from core.trust_result import SignalResult
import math


class ScoringStrategy:
    """
    Estratégias estáticas para agregar scores de múltiplos signals
    """

    @staticmethod
    def weighted_average(
        signal_results: List[SignalResult],
        weights: Dict[str, float]
    ) -> float:
        """
        Média ponderada simples

        Args:
            signal_results: Lista de resultados
            weights: Pesos por signal

        Returns:
            Score agregado [0, 1]
        """
        if not signal_results:
            return 0.5

        total_score = 0.0
        total_weight = 0.0

        for sr in signal_results:
            weight = weights.get(sr.signal_name, 0.0)
            total_score += sr.score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.5

    @staticmethod
    def confidence_weighted(
        signal_results: List[SignalResult],
        weights: Dict[str, float]
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Média ponderada considerando confidence dos signals
        Retorna também confidence interval

        Args:
            signal_results: Lista de resultados
            weights: Pesos por signal

        Returns:
            (score, (lower_bound, upper_bound))
        """
        if not signal_results:
            return 0.5, (0.0, 1.0)

        total_score = 0.0
        total_weight = 0.0
        confidence_sum = 0.0

        for sr in signal_results:
            weight = weights.get(sr.signal_name, 0.0)
            confidence_adjusted_weight = weight * sr.confidence

            total_score += sr.score * confidence_adjusted_weight
            total_weight += confidence_adjusted_weight
            confidence_sum += sr.confidence

        score = total_score / total_weight if total_weight > 0 else 0.5

        # Calcula confidence interval baseado na confidence média
        avg_confidence = confidence_sum / len(signal_results)
        margin = (1.0 - avg_confidence) * 0.2  # Margem proporcional à incerteza

        lower = max(0.0, score - margin)
        upper = min(1.0, score + margin)

        return score, (lower, upper)

    @staticmethod
    def geometric_mean(
        signal_results: List[SignalResult],
        weights: Dict[str, float]
    ) -> float:
        """
        Média geométrica ponderada
        Mais conservadora - um signal baixo puxa o score para baixo

        Args:
            signal_results: Lista de resultados
            weights: Pesos por signal

        Returns:
            Score agregado [0, 1]
        """
        if not signal_results:
            return 0.5

        product = 1.0
        total_weight = sum(weights.get(sr.signal_name, 0.0) for sr in signal_results)

        for sr in signal_results:
            weight = weights.get(sr.signal_name, 0.0)
            normalized_weight = weight / total_weight if total_weight > 0 else 1.0 / len(signal_results)

            # Evita log(0)
            score = max(0.001, sr.score)
            product *= score ** normalized_weight

        return product

    @staticmethod
    def harmonic_mean(
        signal_results: List[SignalResult],
        weights: Dict[str, float]
    ) -> float:
        """
        Média harmônica ponderada
        Ainda mais conservadora que geométrica

        Args:
            signal_results: Lista de resultados
            weights: Pesos por signal

        Returns:
            Score agregado [0, 1]
        """
        if not signal_results:
            return 0.5

        total_weighted_inverse = 0.0
        total_weight = 0.0

        for sr in signal_results:
            weight = weights.get(sr.signal_name, 0.0)
            # Evita divisão por zero
            score = max(0.001, sr.score)

            total_weighted_inverse += weight / score
            total_weight += weight

        if total_weighted_inverse == 0:
            return 0.5

        return total_weight / total_weighted_inverse

    @staticmethod
    def minimum(
        signal_results: List[SignalResult],
        weights: Dict[str, float]
    ) -> float:
        """
        Retorna o menor score (mais conservador)
        Útil para cenários de alta segurança

        Args:
            signal_results: Lista de resultados
            weights: Pesos (não usado, mas mantido para interface consistente)

        Returns:
            Score mínimo [0, 1]
        """
        if not signal_results:
            return 0.5

        return min(sr.score for sr in signal_results)

    @staticmethod
    def adaptive(
        signal_results: List[SignalResult],
        weights: Dict[str, float]
    ) -> float:
        """
        Estratégia adaptativa que escolhe entre média e geométrica
        baseado na variância dos scores

        Args:
            signal_results: Lista de resultados
            weights: Pesos por signal

        Returns:
            Score agregado [0, 1]
        """
        if not signal_results:
            return 0.5

        # Calcula variância
        scores = [sr.score for sr in signal_results]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        # Se variância alta, usa geométrica (mais conservadora)
        # Se variância baixa, usa média ponderada
        if variance > 0.1:
            return ScoringStrategy.geometric_mean(signal_results, weights)
        else:
            return ScoringStrategy.weighted_average(signal_results, weights)

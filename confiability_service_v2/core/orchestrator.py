"""
Trust Orchestrator - Coordena a execução de sinais e agregação
"""
from typing import List
from core.trust_context import TrustContext
from core.trust_result import TrustResult, SignalResult
from aggregation.aggregator import TrustAggregator
from config.trust_config import TrustConfig

# Importar todos os sinais
from signals.temporal import TemporalSignal, TemporalDriftSignal
from signals.source import SourceReliabilitySignal, SourceConsistencySignal
from signals.semantic import SemanticConsistencySignal, SemanticDriftSignal
from signals.anomaly import AnomalyDetectionSignal
from signals.consistency import ConsistencySignal
from signals.context import ContextAlignmentSignal, ContextStabilitySignal

# Storage
from storage.trust_repository import TrustRepository
from storage.fingerprint_repo import FingerprintRepository


class TrustOrchestrator:
    """
    Orquestra a execução de todos os sinais de confiança e agrega os resultados
    """

    def __init__(
            self,
            config: TrustConfig,
            trust_repo: TrustRepository,
            fingerprint_repo: FingerprintRepository
    ):
        self.config = config
        self.trust_repo = trust_repo
        self.fingerprint_repo = fingerprint_repo

        # Inicializar agregador
        self.aggregator = TrustAggregator(config=config)

        # Registrar todos os sinais
        self.signals = self._initialize_signals()

    def _initialize_signals(self):
        """Inicializa todos os sinais disponíveis"""
        return [
            TemporalSignal(),
            TemporalDriftSignal(),
            SourceReliabilitySignal(self.trust_repo),
            SourceConsistencySignal(self.fingerprint_repo),
            SemanticConsistencySignal(self.trust_repo),
            SemanticDriftSignal(self.trust_repo),
            AnomalyDetectionSignal(),
            ConsistencySignal(self.trust_repo),
            ContextAlignmentSignal(),
            ContextStabilitySignal(self.trust_repo),
        ]

    def evaluate(self, context: TrustContext) -> TrustResult:
        """
        Executa todos os sinais habilitados e agrega os resultados

        Args:
            context: Contexto da avaliação de confiança

        Returns:
            TrustResult com score, nível, dimensões, flags e explicações
        """
        signal_results: List[SignalResult] = []

        # Executar cada sinal habilitado
        for signal in self.signals:
            signal_config = self.config.get_signal(signal.name)

            if not signal_config.enabled:
                continue

            try:
                result = signal.evaluate(context, self.config)
                signal_results.append(result)
            except Exception as e:
                # Em caso de erro, criar resultado com score baixo
                signal_results.append(
                    SignalResult(
                        name=signal.name,
                        score=0.0,
                        confidence=0.2,
                        metadata={
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )
                )

        # Agregar resultados
        trust_result = self.aggregator.aggregate(
            context=context,
            signal_results=signal_results
        )

        # Persistir resultado no repositório
        self._persist_result(context, trust_result)

        return trust_result

    def _persist_result(self, context: TrustContext, result: TrustResult):
        """Persiste o resultado da avaliação no repositório"""
        try:
            event = {
                "timestamp": context.timestamp,
                "trust_score": result.trust_score,
                "trust_level": result.trust_level,
                "payload_hash": context.payload_hash,
                "payload_fp": context.payload_fp,
                "dimensions": result.dimensions,
                "risk_flags": result.risk_flags
            }

            # Salvar por entidade
            if context.entity_id:
                self.trust_repo.add_event(
                    key=f"entity:{context.entity_id}",
                    event=event
                )

            # Salvar por source
            if context.source_id:
                self.trust_repo.add_event(
                    key=f"source:{context.source_id}",
                    event=event
                )

            # Salvar fingerprint
            if context.source_id and context.payload_fp:
                self.fingerprint_repo.add_fingerprint(
                    source_id=context.source_id,
                    fingerprint=context.payload_fp,
                    timestamp=context.timestamp
                )
        except Exception as e:
            # Não falhar a avaliação por erro de persistência
            pass
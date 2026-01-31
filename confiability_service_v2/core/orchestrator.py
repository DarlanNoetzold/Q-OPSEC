"""
Trust Orchestrator - Coordena a avaliação de confiança
Gerencia signals, agregação e geração de resultados
"""
from typing import List
from core.trust_context import TrustContext
from core.trust_result import TrustResult
from config.trust_config import TrustConfig
from storage.trust_repository import TrustRepository
from storage.fingerprint_repo import FingerprintRepository
from signals.base import TrustSignal
from aggregation.aggregator import TrustAggregator

# Importar todos os signals
from signals.temporal import TemporalSignal, TemporalDriftSignal
from signals.source import SourceReliabilitySignal, SourceConsistencySignal
from signals.semantic import SemanticConsistencySignal, SemanticDriftSignal
from signals.anomaly import AnomalyDetectionSignal
from signals.consistency import ConsistencySignal
from signals.context import ContextAlignmentSignal, ContextStabilitySignal


class TrustOrchestrator:
    """
    Orquestrador principal do Trust Engine V2
    Coordena signals, agregação e persistência
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

        # Inicializar signals (passando config e trust_repo)
        self.signals = self._initialize_signals()

        # Inicializar agregador
        self.aggregator = TrustAggregator(
            config=config,
            trust_repo=trust_repo
        )

    def _initialize_signals(self) -> List[TrustSignal]:
        """
        Inicializa todos os signals disponíveis
        Cada signal recebe config e trust_repo
        """
        return [
            # Temporal
            TemporalSignal(config=self.config, trust_repo=self.trust_repo),
            TemporalDriftSignal(config=self.config, trust_repo=self.trust_repo),

            # Source
            SourceReliabilitySignal(config=self.config, trust_repo=self.trust_repo),
            SourceConsistencySignal(config=self.config, trust_repo=self.trust_repo),

            # Semantic
            SemanticConsistencySignal(config=self.config, trust_repo=self.trust_repo),
            SemanticDriftSignal(config=self.config, trust_repo=self.trust_repo),

            # Anomaly
            AnomalyDetectionSignal(config=self.config, trust_repo=self.trust_repo),

            # Consistency
            ConsistencySignal(config=self.config, trust_repo=self.trust_repo),

            # Context
            ContextAlignmentSignal(config=self.config, trust_repo=self.trust_repo),
            ContextStabilitySignal(config=self.config, trust_repo=self.trust_repo),
        ]

    def evaluate(self, context: TrustContext) -> TrustResult:
        """
        Executa avaliação completa de confiança

        1. Executa todos os signals
        2. Agrega resultados
        3. Persiste histórico
        4. Retorna resultado final
        """
        # 1. Executar todos os signals
        signal_results = []
        for signal in self.signals:
            try:
                result = signal.evaluate(context)
                signal_results.append(result)
            except Exception as e:
                # Log error mas continua com outros signals
                print(f"⚠️  Signal {signal.name} failed: {e}")
                # Criar resultado de fallback
                from core.trust_result import SignalResult
                signal_results.append(
                    SignalResult(
                        signal_name=signal.name,
                        score=0.5,  # Neutro
                        confidence=0.0,
                        metadata={"error": str(e), "status": "failed"}
                    )
                )

        # 2. Agregar resultados
        trust_result = self.aggregator.aggregate(
            context=context,
            signal_results=signal_results
        )

        # 3. Persistir no histórico
        self._persist_evaluation(context, trust_result)

        return trust_result

    def _persist_evaluation(self, context: TrustContext, result: TrustResult):
        """
        Persiste avaliação no histórico
        """
        try:
            # Criar evento de histórico
            event = {
                "timestamp": context.timestamp,
                "payload_hash": context.payload_hash,
                "payload_fp": context.payload_fp,
                "trust_score": result.trust_score,
                "trust_level": result.trust_level,
                "dimensions": result.dimensions,
                "risk_flags": result.risk_flags,
                "metadata": context.metadata
            }

            # Adicionar ao repositório
            if context.entity_id:
                self.trust_repo.add_entity_event(context.entity_id, event)

            if context.source_id:
                self.trust_repo.add_source_event(context.source_id, event)

            # Adicionar fingerprint
            if context.payload_fp:
                self.fingerprint_repo.add_fingerprint(
                    source_id=context.source_id or "unknown",
                    fingerprint=context.payload_fp,
                    metadata={
                        "entity_id": context.entity_id,
                        "timestamp": context.timestamp,
                        "trust_score": result.trust_score
                    }
                )

        except Exception as e:
            # Não falhar a avaliação se persistência falhar
            print(f"⚠️  Failed to persist evaluation: {e}")

    def get_stats(self) -> dict:
        """
        Retorna estatísticas do engine
        """
        return {
            "signals": {
                "total": len(self.signals),
                "names": [s.name for s in self.signals]
            },
            "repository": {
                "entities": len(self.trust_repo.entity_history),
                "sources": len(self.trust_repo.source_history),
                "total_events": sum(
                    len(events) for events in self.trust_repo.entity_history.values()
                ) + sum(
                    len(events) for events in self.trust_repo.source_history.values()
                )
            },
            "config": {
                "trust_levels": self.config.trust_levels,
                "max_history_size": self.config.max_history_size
            }
        }
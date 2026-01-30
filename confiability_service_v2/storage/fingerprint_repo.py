"""
Fingerprint Repository - Armazena e analisa fingerprints estruturais de payloads
"""
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime
import threading


class FingerprintRepository:
    """
    Repositório especializado para fingerprints de payloads.
    Permite detectar padrões estruturais e anomalias.
    """

    def __init__(self):
        # source_id -> [{"fp": str, "timestamp": datetime, "count": int}]
        self._fingerprints: Dict[str, List[dict]] = defaultdict(list)

        # fingerprint -> {"sources": set, "first_seen": datetime, "last_seen": datetime, "count": int}
        self._fp_index: Dict[str, dict] = {}

        self._lock = threading.Lock()

    def add_fingerprint(
            self,
            source_id: str,
            fingerprint: str,
            timestamp: datetime
    ):
        """
        Adiciona um fingerprint ao repositório

        Args:
            source_id: ID da fonte
            fingerprint: Hash estrutural do payload
            timestamp: Momento da captura
        """
        with self._lock:
            # Adicionar ao índice por source
            existing = next(
                (fp for fp in self._fingerprints[source_id] if fp["fp"] == fingerprint),
                None
            )

            if existing:
                existing["count"] += 1
                existing["last_seen"] = timestamp
            else:
                self._fingerprints[source_id].append({
                    "fp": fingerprint,
                    "timestamp": timestamp,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "count": 1
                })

            # Adicionar ao índice global
            if fingerprint not in self._fp_index:
                self._fp_index[fingerprint] = {
                    "sources": set(),
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "count": 0
                }

            self._fp_index[fingerprint]["sources"].add(source_id)
            self._fp_index[fingerprint]["last_seen"] = max(
                self._fp_index[fingerprint]["last_seen"],
                timestamp
            )
            self._fp_index[fingerprint]["count"] += 1

    def get_source_fingerprints(
            self,
            source_id: str,
            limit: int = 100
    ) -> List[dict]:
        """
        Retorna fingerprints de uma fonte específica

        Args:
            source_id: ID da fonte
            limit: Número máximo de fingerprints

        Returns:
            Lista de fingerprints ordenados por timestamp (mais recentes primeiro)
        """
        with self._lock:
            fps = self._fingerprints.get(source_id, [])
            sorted_fps = sorted(fps, key=lambda x: x["last_seen"], reverse=True)
            return sorted_fps[:limit]

    def get_fingerprint_info(self, fingerprint: str) -> Optional[dict]:
        """
        Retorna informações sobre um fingerprint específico

        Args:
            fingerprint: Hash estrutural

        Returns:
            Dicionário com informações ou None se não encontrado
        """
        with self._lock:
            info = self._fp_index.get(fingerprint)
            if info:
                return {
                    "fingerprint": fingerprint,
                    "sources": list(info["sources"]),
                    "source_count": len(info["sources"]),
                    "first_seen": info["first_seen"],
                    "last_seen": info["last_seen"],
                    "total_count": info["count"]
                }
            return None

    def calculate_source_consistency(
            self,
            source_id: str,
            current_fp: str
    ) -> float:
        """
        Calcula consistência estrutural de uma fonte

        Args:
            source_id: ID da fonte
            current_fp: Fingerprint atual

        Returns:
            Score de consistência (0-1)
        """
        fps = self.get_source_fingerprints(source_id, limit=50)

        if not fps:
            return 0.5  # Neutro para fontes novas

        # Verificar se o fingerprint atual já foi visto
        fp_counts = {fp["fp"]: fp["count"] for fp in fps}
        total_count = sum(fp_counts.values())

        if current_fp in fp_counts:
            # Fingerprint conhecido - alta consistência
            frequency = fp_counts[current_fp] / total_count
            return 0.7 + (frequency * 0.3)  # 0.7 - 1.0

        # Fingerprint novo
        unique_fps = len(fp_counts)

        if unique_fps < 3:
            # Fonte com poucos padrões - fingerprint novo é suspeito
            return 0.4
        elif unique_fps < 10:
            # Fonte com variação moderada - aceitável
            return 0.6
        else:
            # Fonte com muita variação - fingerprint novo é normal
            return 0.7

    def detect_fingerprint_anomalies(
            self,
            source_id: str,
            fingerprint: str
    ) -> List[str]:
        """
        Detecta anomalias no fingerprint

        Args:
            source_id: ID da fonte
            fingerprint: Fingerprint a analisar

        Returns:
            Lista de anomalias detectadas
        """
        anomalies = []

        # Verificar se fingerprint é usado por múltiplas fontes (possível cópia)
        fp_info = self.get_fingerprint_info(fingerprint)
        if fp_info and fp_info["source_count"] > 5:
            anomalies.append("fingerprint_shared_across_sources")

        # Verificar variação estrutural da fonte
        source_fps = self.get_source_fingerprints(source_id, limit=20)
        if len(source_fps) > 10:
            unique_fps = len(set(fp["fp"] for fp in source_fps))
            if unique_fps > 15:
                anomalies.append("high_structural_variability")

        return anomalies

    def get_similar_sources(
            self,
            source_id: str,
            min_shared_fps: int = 3
    ) -> List[tuple[str, int]]:
        """
        Encontra fontes com fingerprints similares

        Args:
            source_id: ID da fonte
            min_shared_fps: Mínimo de fingerprints compartilhados

        Returns:
            Lista de (source_id, shared_count) ordenada por similaridade
        """
        source_fps = self.get_source_fingerprints(source_id)
        source_fp_set = set(fp["fp"] for fp in source_fps)

        similar_sources = defaultdict(int)

        for fp in source_fp_set:
            fp_info = self.get_fingerprint_info(fp)
            if fp_info:
                for other_source in fp_info["sources"]:
                    if other_source != source_id:
                        similar_sources[other_source] += 1

        # Filtrar e ordenar
        result = [
            (src, count)
            for src, count in similar_sources.items()
            if count >= min_shared_fps
        ]

        return sorted(result, key=lambda x: x[1], reverse=True)

    def clear(self):
        """Limpa todos os dados (útil para testes)"""
        with self._lock:
            self._fingerprints.clear()
            self._fp_index.clear()
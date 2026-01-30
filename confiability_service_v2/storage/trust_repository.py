"""
Trust repository - stores and retrieves trust evaluation history.
In-memory implementation (can be replaced with database).
"""
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict, deque
from core.trust_context import TrustContext
from core.trust_result import TrustResult
from utils.time import now_utc, parse_iso
import threading


class TrustRepository:
    """
    In-memory storage for trust evaluation history.
    Thread-safe implementation.
    """

    def __init__(self, max_history_per_entity: int = 100):
        """
        Initialize repository.

        Args:
            max_history_per_entity: Maximum events to store per entity
        """
        self.max_history = max_history_per_entity

        # Storage structures
        self._entity_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        self._source_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        self._source_fingerprints: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )

        # Thread safety
        self._lock = threading.RLock()

    def store(self, context: TrustContext, result: TrustResult):
        """
        Store trust evaluation result.

        Args:
            context: Trust context
            result: Trust result
        """
        with self._lock:
            event = self._create_event(context, result)

            # Store by entity
            if context.entity_id:
                self._entity_history[context.entity_id].append(event)

            # Store by source
            if context.source_id:
                self._source_history[context.source_id].append(event)

                # Store fingerprint
                fp_event = {
                    "timestamp": context.timestamp.isoformat(),
                    "payload_fp": context.payload_fp,
                    "trust_score": result.trust_score
                }
                self._source_fingerprints[context.source_id].append(fp_event)

    def get_entity_history(
            self,
            entity_id: str,
            data_type: Optional[str] = None,
            limit: int = 10
    ) -> List[Dict]:
        """
        Get historical events for an entity.

        Args:
            entity_id: Entity identifier
            data_type: Optional filter by data type
            limit: Maximum number of events to return

        Returns:
            List of historical events (most recent first)
        """
        with self._lock:
            if entity_id not in self._entity_history:
                return []

            history = list(self._entity_history[entity_id])

            # Filter by data type if specified
            if data_type:
                history = [e for e in history if e.get("data_type") == data_type]

            # Return most recent first
            return list(reversed(history))[:limit]

    def get_source_history(
            self,
            source_id: str,
            limit: int = 10
    ) -> List[Dict]:
        """
        Get historical events for a source.

        Args:
            source_id: Source identifier
            limit: Maximum number of events to return

        Returns:
            List of historical events (most recent first)
        """
        with self._lock:
            if source_id not in self._source_history:
                return []

            history = list(self._source_history[source_id])
            return list(reversed(history))[:limit]

    def get_source_fingerprints(
            self,
            source_id: str,
            limit: int = 10
    ) -> List[Dict]:
        """
        Get fingerprint history for a source.

        Args:
            source_id: Source identifier
            limit: Maximum number of fingerprints to return

        Returns:
            List of fingerprint events
        """
        with self._lock:
            if source_id not in self._source_fingerprints:
                return []

            fps = list(self._source_fingerprints[source_id])
            return list(reversed(fps))[:limit]

    def get_last_event(
            self,
            entity_id: Optional[str] = None,
            source_id: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get the most recent event for entity or source.

        Args:
            entity_id: Optional entity identifier
            source_id: Optional source identifier

        Returns:
            Most recent event or None
        """
        with self._lock:
            if entity_id and entity_id in self._entity_history:
                history = self._entity_history[entity_id]
                if history:
                    return history[-1]

            if source_id and source_id in self._source_history:
                history = self._source_history[source_id]
                if history:
                    return history[-1]

            return None

    def get_stats(self) -> Dict:
        """
        Get repository statistics.

        Returns:
            Dictionary with stats
        """
        with self._lock:
            return {
                "total_entities": len(self._entity_history),
                "total_sources": len(self._source_history),
                "total_events": sum(
                    len(history) for history in self._entity_history.values()
                )
            }

    def clear(self):
        """Clear all stored data."""
        with self._lock:
            self._entity_history.clear()
            self._source_history.clear()
            self._source_fingerprints.clear()

    def _create_event(self, context: TrustContext, result: TrustResult) -> Dict:
        """Create event dictionary from context and result."""
        return {
            "timestamp": context.timestamp.isoformat(),
            "source_id": context.source_id,
            "entity_id": context.entity_id,
            "data_type": context.data_type,
            "environment": context.environment,
            "payload_hash": context.payload_hash,
            "payload_fp": context.payload_fp,
            "trust_score": result.trust_score,
            "trust_level": result.trust_level,
            "trust_dna": result.trust_dna_value,
            "payload": context.payload  # Store for semantic analysis
        }
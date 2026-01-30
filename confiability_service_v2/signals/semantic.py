"""
Semantic consistency signal - detects contradictions and drift in information.
"""
from signals.base import TrustSignal
from core.trust_context import TrustContext
from core.trust_result import SignalResult
from utils.hashing import stable_hash
from typing import List, Dict, Any
import json


class SemanticConsistencySignal(TrustSignal):
    """
    Evaluates semantic consistency with historical information.
    Detects contradictions and narrative drift.
    """

    @property
    def name(self) -> str:
        return "semantic_consistency"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """
        Compare current payload with historical payloads for consistency.
        """
        entity_id = context.entity_id

        if not entity_id or not self.storage:
            return self._create_result(0.5, {"reason": "no_history"})

        # Get historical payloads
        history = self.storage.get_entity_history(
            entity_id=entity_id,
            data_type=context.data_type,
            limit=self.params.get("max_history", 10)
        )

        if not history:
            return self._create_result(0.5, {"reason": "first_event"})

        # Extract key-value pairs for comparison
        current_claims = self._extract_claims(context.payload)

        contradictions = []
        confirmations = []

        for event in history:
            historical_payload = event.get("payload", {})
            historical_claims = self._extract_claims(historical_payload)

            # Compare claims
            for key, current_value in current_claims.items():
                if key in historical_claims:
                    historical_value = historical_claims[key]

                    if self._are_contradictory(current_value, historical_value):
                        contradictions.append({
                            "key": key,
                            "current": str(current_value)[:50],
                            "historical": str(historical_value)[:50]
                        })
                    elif current_value == historical_value:
                        confirmations.append(key)

        # Calculate score
        total_comparisons = len(contradictions) + len(confirmations)

        if total_comparisons == 0:
            score = 0.7  # no overlap = neutral-positive
        else:
            contradiction_rate = len(contradictions) / total_comparisons
            score = 1.0 - (contradiction_rate * 1.5)  # heavy penalty

        metadata = {
            "contradictions": len(contradictions),
            "confirmations": len(confirmations),
            "contradiction_details": contradictions[:3],  # top 3
            "historical_events_checked": len(history)
        }

        return self._create_result(score, metadata)

    def _extract_claims(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key claims from payload.
        Flattens nested structures.
        """
        claims = {}

        def flatten(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, (dict, list)):
                        flatten(v, new_key)
                    else:
                        claims[new_key] = v
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    flatten(item, f"{prefix}[{i}]")

        flatten(payload)
        return claims

    def _are_contradictory(self, val1: Any, val2: Any) -> bool:
        """
        Check if two values are contradictory.
        """
        # Same value = not contradictory
        if val1 == val2:
            return False

        # Different types = potentially contradictory
        if type(val1) != type(val2):
            return True

        # For strings, check if they're very different
        if isinstance(val1, str) and isinstance(val2, str):
            # Simple heuristic: if both non-empty and different, it's a contradiction
            if val1.strip() and val2.strip() and val1.lower() != val2.lower():
                return True

        # For numbers, check if difference is significant
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if abs(val1 - val2) > 0.01:  # threshold
                return True

        return False


class SemanticDriftSignal(TrustSignal):
    """
    Measures how much an entity's information "drifts" over time.
    """

    @property
    def name(self) -> str:
        return "semantic_drift"

    def evaluate(self, context: TrustContext) -> SignalResult:
        """
        Calculate semantic drift index (SDI).
        """
        entity_id = context.entity_id

        if not entity_id or not self.storage:
            return self._create_result(1.0, {"reason": "no_history"})

        history = self.storage.get_entity_history(
            entity_id=entity_id,
            limit=5
        )

        if len(history) < 2:
            return self._create_result(1.0, {"reason": "insufficient_history"})

        # Calculate hash similarity over time
        current_hash = context.payload_hash
        historical_hashes = [event.get("payload_hash") for event in history]

        # Count unique hashes (more unique = more drift)
        unique_hashes = len(set(historical_hashes + [current_hash]))
        total_events = len(historical_hashes) + 1

        drift_rate = unique_hashes / total_events

        # Low drift = high score
        score = 1.0 - (drift_rate * 0.5)  # max penalty 0.5

        metadata = {
            "drift_rate": round(drift_rate, 4),
            "unique_versions": unique_hashes,
            "total_events": total_events
        }

        return self._create_result(score, metadata)
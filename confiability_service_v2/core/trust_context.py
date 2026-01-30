"""
Trust evaluation context - encapsulates all information needed for trust assessment.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from utils.time import now_utc, parse_iso
from utils.hashing import stable_hash, payload_fingerprint


@dataclass
class TrustContext:
    """
    Immutable context for trust evaluation.
    Contains payload, metadata, and derived properties.
    """

    # Core data
    payload: Dict[str, Any]
    metadata: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)

    # Derived properties (computed on init)
    timestamp: datetime = field(init=False)
    source_id: Optional[str] = field(init=False)
    entity_id: Optional[str] = field(init=False)
    data_type: Optional[str] = field(init=False)
    environment: Optional[str] = field(init=False)
    request_id: Optional[str] = field(init=False)

    # Fingerprints
    payload_hash: str = field(init=False)
    payload_fp: str = field(init=False)

    def __post_init__(self):
        """Extract and compute derived properties."""
        # Extract metadata
        self.source_id = self.metadata.get("source_id")
        self.entity_id = self.metadata.get("entity_id")
        self.data_type = self.metadata.get("data_type", "unknown")

        # Parse timestamp
        ts_str = self.metadata.get("timestamp")
        if ts_str:
            self.timestamp = parse_iso(ts_str)
        else:
            self.timestamp = now_utc()

        # Extract context
        self.environment = self.context.get("environment", "unknown")
        self.request_id = self.context.get("request_id")

        # Generate fingerprints
        self.payload_hash = stable_hash(self.payload)
        self.payload_fp = payload_fingerprint(self.payload)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "payload_hash": self.payload_hash,
            "payload_fp": self.payload_fp,
            "source_id": self.source_id,
            "entity_id": self.entity_id,
            "data_type": self.data_type,
            "environment": self.environment,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id
        }

    def __repr__(self) -> str:
        return (f"TrustContext(source={self.source_id}, "
                f"entity={self.entity_id}, "
                f"type={self.data_type}, "
                f"ts={self.timestamp.isoformat()})")
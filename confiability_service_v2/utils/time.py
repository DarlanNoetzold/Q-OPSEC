"""
Temporal utilities for trust evaluation.
"""
from datetime import datetime, timezone
from typing import Optional


def now_utc() -> datetime:
    """Returns current UTC datetime."""
    return datetime.now(timezone.utc)


def parse_iso(timestamp: str) -> datetime:
    """Parse ISO 8601 timestamp to datetime."""
    return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))


def delta_seconds(dt1: datetime, dt2: datetime) -> float:
    """Calculate time difference in seconds."""
    return abs((dt1 - dt2).total_seconds())


def delta_hours(dt1: datetime, dt2: datetime) -> float:
    """Calculate time difference in hours."""
    return delta_seconds(dt1, dt2) / 3600


def delta_days(dt1: datetime, dt2: datetime) -> float:
    """Calculate time difference in days."""
    return delta_seconds(dt1, dt2) / 86400


def is_expired(timestamp: datetime, max_age_seconds: float) -> bool:
    """Check if timestamp is older than max_age_seconds."""
    return delta_seconds(now_utc(), timestamp) > max_age_seconds
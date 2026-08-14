"""
Metrics and monitoring service.
"""
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from threading import Lock

from ..core.logging import get_logger

logger = get_logger(__name__)


class MetricsService:
    """Service for collecting and managing API metrics."""

    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.start_time = time.time()
        self.lock = Lock()

        # Counters
        self.total_requests = 0
        self.total_predictions = 0
        self.total_errors = 0
        self.model_reload_count = 0

        # Response times (sliding window)
        self.response_times = deque(maxlen=max_history_size)

        # Error tracking
        self.error_counts = defaultdict(int)

        # Request tracking
        self.requests_by_endpoint = defaultdict(int)
        self.requests_by_status = defaultdict(int)

        # Prediction tracking
        self.predictions_by_model = defaultdict(int)
        self.last_prediction_at: Optional[datetime] = None

        # Current model
        self.current_model: Optional[str] = None

    def record_request(self, endpoint: str, method: str, status_code: int, response_time: float):
        """Record a request with its metrics."""
        with self.lock:
            self.total_requests += 1
            self.response_times.append(response_time)
            self.requests_by_endpoint[f"{method} {endpoint}"] += 1
            self.requests_by_status[status_code] += 1

            if status_code >= 400:
                self.total_errors += 1

    def record_prediction(self, model_name: str, batch_size: int = 1):
        """Record a prediction event."""
        with self.lock:
            self.total_predictions += batch_size
            self.predictions_by_model[model_name] += batch_size
            self.last_prediction_at = datetime.utcnow()
            self.current_model = model_name

    def record_error(self, error_type: str):
        """Record an error event."""
        with self.lock:
            self.error_counts[error_type] += 1

    def record_model_reload(self, model_name: str):
        """Record a model reload event."""
        with self.lock:
            self.model_reload_count += 1
            self.current_model = model_name

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        with self.lock:
            uptime = time.time() - self.start_time

            # Calculate average response time
            avg_response_time = 0.0
            if self.response_times:
                avg_response_time = sum(self.response_times) / len(self.response_times) * 1000  # Convert to ms

            # Calculate error rate
            error_rate = 0.0
            if self.total_requests > 0:
                error_rate = self.total_errors / self.total_requests

            return {
                "total_requests": self.total_requests,
                "total_predictions": self.total_predictions,
                "total_errors": self.total_errors,
                "average_response_time_ms": round(avg_response_time, 2),
                "error_rate": round(error_rate, 4),
                "model_reload_count": self.model_reload_count,
                "uptime_seconds": round(uptime, 2),
                "last_prediction_at": self.last_prediction_at,
                "current_model": self.current_model,
                "requests_by_endpoint": dict(self.requests_by_endpoint),
                "requests_by_status": dict(self.requests_by_status),
                "predictions_by_model": dict(self.predictions_by_model),
                "error_counts": dict(self.error_counts)
            }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status based on metrics."""
        with self.lock:
            uptime = time.time() - self.start_time

            # Determine health status
            status = "healthy"
            issues = []

            # Check error rate
            if self.total_requests > 10:  # Only check if we have enough requests
                error_rate = self.total_errors / self.total_requests
                if error_rate > 0.1:  # More than 10% errors
                    status = "degraded"
                    issues.append(f"High error rate: {error_rate:.2%}")

            # Check if model is loaded
            if not self.current_model:
                status = "degraded"
                issues.append("No model loaded")

            # Check recent activity (if we've had predictions)
            if self.last_prediction_at:
                time_since_last = datetime.utcnow() - self.last_prediction_at
                if time_since_last > timedelta(hours=1):
                    # This might be normal, so just note it
                    issues.append(f"No predictions in {time_since_last}")

            return {
                "status": status,
                "uptime_seconds": round(uptime, 2),
                "issues": issues,
                "model_loaded": bool(self.current_model),
                "model_name": self.current_model
            }

    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        with self.lock:
            self.total_requests = 0
            self.total_predictions = 0
            self.total_errors = 0
            self.model_reload_count = 0
            self.response_times.clear()
            self.error_counts.clear()
            self.requests_by_endpoint.clear()
            self.requests_by_status.clear()
            self.predictions_by_model.clear()
            self.last_prediction_at = None
            self.current_model = None
            self.start_time = time.time()


# Global metrics service instance
metrics_service = MetricsService()
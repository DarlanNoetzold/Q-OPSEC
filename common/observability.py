from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def instrument_app(app: Any) -> bool:
    """Expose Prometheus /metrics on a FastAPI app; monitoring failures never break startup."""
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app)
        return True
    except Exception as exc:
        logger.warning("Monitoring disabled: %s", exc)
        return False

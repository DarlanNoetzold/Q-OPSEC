"""Optional OpenTelemetry wiring with a no-op fallback.

The packages are optional: when they are absent the helpers degrade to a no-op so
services keep booting on a machine without the tracing stack installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("qopsec.tracing")

_configured = False
_enabled = False


def tracing_enabled() -> bool:
    return _enabled


def setup_tracing(
    app: Any = None,
    service_name: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    sample_ratio: Optional[float] = None,
) -> bool:
    """Instrument a FastAPI app and outgoing httpx calls when OTel is available."""
    global _configured, _enabled
    if _configured:
        return _enabled
    _configured = True

    if os.environ.get("QOPSEC_TRACING_ENABLED", "auto").strip().lower() in {"0", "false", "off", "no"}:
        logger.info("Tracing disabled by QOPSEC_TRACING_ENABLED")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    except Exception:
        logger.info("OpenTelemetry SDK not installed; tracing stays disabled")
        return False

    name = service_name or os.environ.get("QOPSEC_SERVICE_NAME", "qopsec")
    ratio = float(sample_ratio if sample_ratio is not None else os.environ.get("QOPSEC_TRACE_SAMPLE_RATIO", "0.1"))
    provider = TracerProvider(
        resource=Resource.create({"service.name": name}),
        sampler=ParentBased(TraceIdRatioBased(max(0.0, min(1.0, ratio)))),
    )

    endpoint = otlp_endpoint or os.environ.get("QOPSEC_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        except Exception as exc:
            logger.warning("OTLP exporter unavailable (%s); spans stay in-process", type(exc).__name__)

    trace.set_tracer_provider(provider)

    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app, excluded_urls="/health,/architecture-metrics,/metrics")
        except Exception as exc:
            logger.warning("FastAPI instrumentation unavailable (%s)", type(exc).__name__)
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    except Exception as exc:
        logger.warning("httpx instrumentation unavailable (%s)", type(exc).__name__)

    _enabled = True
    logger.info("Tracing enabled for %s (sample ratio %.2f)", name, ratio)
    return True


def current_trace_id() -> Optional[str]:
    if not _enabled:
        return None
    try:
        from opentelemetry import trace

        span_context = trace.get_current_span().get_span_context()
        if not span_context.is_valid:
            return None
        return format(span_context.trace_id, "032x")
    except Exception:
        return None


def trace_headers() -> Dict[str, str]:
    trace_id = current_trace_id()
    return {"X-Trace-Id": trace_id} if trace_id else {}

"""Optional OpenTelemetry setup for the inference service."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Mapping


@contextmanager
def request_span(name: str, headers: Mapping[str, str]) -> Iterator[object | None]:
    """Create a server span when the API image's OTel dependencies are present."""
    try:
        from opentelemetry import propagate, trace
        tracer = trace.get_tracer("pw2048.api")
        context = propagate.extract(headers)
        with tracer.start_as_current_span(name, context=context) as span:
            yield span
    except ImportError:
        yield None


def configure_tracing(release: str, model: str, version: str) -> None:
    """Configure OTLP/HTTP export only when an endpoint is supplied."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({
        "service.name": "pw2048-inference", "deployment.environment": release,
        "pw2048.model.name": model, "pw2048.model.version": version,
    }))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")))
    trace.set_tracer_provider(provider)

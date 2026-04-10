"""
Metrics and Observability

Provides OpenTelemetry traces and metrics for agent-brain operations.
Zero overhead if no exporter is configured by the host application.
"""

import time
from functools import wraps
from opentelemetry import trace, metrics

# Acquire telemetry components
tracer = trace.get_tracer("agent_brain.tracer")
meter = metrics.get_meter("agent_brain.meter")

# Define metrics
query_latency = meter.create_histogram(
    "agent_brain.query.latency",
    unit="ms",
    description="Latency of Neo4jBrain queries and embeddings"
)

error_counter = meter.create_counter(
    "agent_brain.errors",
    description="Counts of errors raised by agent_brain"
)

memory_op_counter = meter.create_counter(
    "agent_brain.memory_operations",
    description="Counts of memory additions, archives, and consolidations"
)


def trace_and_measure(op_name: str, is_memory_op: bool = False):
    """
    Decorator to wrap brain methods with OpenTelemetry traces and metrics.
    Records duration in `agent_brain.query.latency` and increments error counters.
    Optionally increments `agent_brain.memory_operations` if `is_memory_op=True`.

    Args:
        op_name: Diagnostic name of the operation (e.g., "semantic_search")
        is_memory_op: Whether this is a counting operation (add, archive)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            with tracer.start_as_current_span(op_name) as span:
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start) * 1000.0
                    
                    # Record metric
                    query_latency.record(duration_ms, {"operation": op_name, "status": "success"})
                    
                    if is_memory_op:
                        memory_op_counter.add(1, {"operation": op_name})

                    return result
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000.0
                    query_latency.record(duration_ms, {"operation": op_name, "status": "error"})
                    
                    error_type = type(e).__name__
                    error_counter.add(1, {"operation": op_name, "error_type": error_type})
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator

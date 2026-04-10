# Scaling `agent-brain` for Production

The `agent-brain` library is designed to gracefully handle significant loads when tuned appropriately. Our architecture isolates embedding calculations, leverages recursive circuit breakers, and introduces temporal memory consolidation.

If you are moving past local development into a high-scale distributed environment, please review these architectural strategies.

---

## 1. Environment and Resource Governance

### Embedding Latency Optimizations
Out of the box, `agent-brain` supports local `sentence-transformers`, APIs like Anthropic/OpenAI, and local API servers like Ollama. In production:
- High volume endpoints benefit significantly by moving to dedicated model clusters to avoid CPU-bound blocking threads.
- Set `AGENT_BRAIN_EMBEDDING_PROVIDER=openai` (or relevant) and construct the connection with `Neo4jBrain(preload_embeddings=True)`. This automatically calls the `.warmup()` function on the singleton provider.

### Database Connection Pooling
The Neo4j Driver natively supports connection pooling. This defaults to 50 concurrent connections, which handles intense spikes well. When hooking into highly-concurrent ASGI frameworks (FastAPI/Tornado), scale this dynamically via your environment:

```bash
export NEO4J_MAX_CONNECTION_POOL_SIZE=150
```

---

## 2. Telemetry and Observability

`agent-brain` includes OpenTelemetry trace limits and counters natively via `opentelemetry-api`. We exclusively utilize the vendor-neutral API, meaning **this operates with zero-overhead if no SDK is configured**.

### Hooking into Datadog / Prometheus
To capture latency profiles of `semantic_search()` and operation counts, configure a meter provider in your app entrypoint:

```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader

# Bootstrap your SDK
reader = PrometheusMetricReader()
provider = MeterProvider(metric_readers=[reader])
metrics.set_meter_provider(provider)

# Later, initialize brain: its metrics will automatically flow downstream!
from agent_brain.brain import Neo4jBrain
```

**Metric Hooks Formatted:**
- `agent_brain.query.latency` (Histogram, ms) — Traces Graph Traversal and Search bottlenecks.
- `agent_brain.memory_operations` (Counter) — Generic metric on persistence scale.
- `agent_brain.errors` (Counter) — Catch sudden spikes that indicate DB network drops or model fallback thresholds.

*To probe health for heartbeat/liveness checks (e.g. k8s readiness probe), return the payload of `Neo4jBrain().get_health()`.*

---

## 3. Database Constraints and Graph Limits

By default, we automatically trigger synchronous configuration matching across the DB on start:
- Unique Node ID Constraints
- Vector indexes dimensionally mapped against your embedding context space (e.g. 1536d)
- BM25 fulltext fallback nodes 

If your topology accumulates millions of nodes, you should monitor vector index metrics natively inside Neo4j's UI. For truly massive graphs, disabling `.ensure_schema()` and managing those explicitly under Terraform is advantageous.

---

## 4. Tuning the Circuit-Breakers

By default, `spread_activation` will recursively execute up to 100 activation hops dynamically. For heavily entwined networks (vast semantic overlap bounded to the same user or container), you will want to constrain this to guarantee uniform REST response times.

```python
results = brain.spread_activation(
    start_entities=["Project A"], 
    iterations=2,             # Shorten theoretical recursive cycles
    decay_factor=0.8,         # Expire recursive importance faster
    max_activated_nodes=35    # Strict ceiling
)
```

---

## 5. Unbounded Memory Consolidation

A core tenet of `agent-brain` is "context rot" suppression. `agent_brain.consolidation.MemoryConsolidator` summarizes historical interaction records over time into concise knowledge.

In production, **never invoke this synchronously in the main web thread.** It generates clustered embeddings via an LLM map!
Instead, run it asynchronously via Airflow, Celery, or a sidecar Cronjob:

```python
from agent_brain.consolidation import MemoryConsolidator

def batch_cron():
    brain = Neo4jBrain()
    consolidator = MemoryConsolidator(brain=brain)
    
    # Executed nightly at 2:00 AM
    stats = consolidator.consolidate_old_memories(
        older_than_days=14,
        importance_threshold=0.85
    )
    print(f"Archived {stats.memories_archived} episodic memories.")
```

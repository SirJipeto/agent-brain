# Changelog

All notable changes to this project will be documented in this file.

## [0.2.1] - 2026-04-11

### Fixed
- Fixed bug where `agent_brain.__version__` misaligned with `setup.py`.
- **Thread Safety:** Made `BrainAgent` explicitly thread-safe using `threading.Lock`.
- **Driver Leakage:** Patched `BrainAgent` initialization to natively wrap the shared `get_connection()` pool instead of spinning headless connection drivers.
- **Dead Code Cleanup:** Removed duplicate decay methodologies and unused structure nodes from core classes.

## [0.2.0] - 2026-04-10

### Added
- **Pluggable Embedding Providers:** Support for SentenceTransformers, OpenAI, and Ollama out of the box with `pip install agent-brain[local|openai|ollama]`.
- **OpenTelemetry Observability:** Zero-overhead tracing of query latency, memory op errors, and volume. Available via `metrics.py`.
- **BM25 Native Indexing:** Uses Neo4j native fulltext indexing for text search fallback when vector similarity is unviable.
- **Dynamic Relationship Discovery:** Custom relationship types are mapped into Neo4j Config nodes dynamically via `auto_discover=True`.
- **Temporal Memory Consolidation:** Memories are now accurately bucketed chronologically via week markers to stabilize graph growth.
- **Production Examples:** Wrappers for FastAPI, LangChain's `BaseTool`, and the Claude API included in `examples/`.

### Changed
- **Relationship Weights:** Default strength of `relate_entities()` dropped from 0.5 to 0.3 to enforce stricter propagation decay.
- **Age-Decay Traversal Search:** Graph edge traversal now natively incorporates a time-decay penalty into path weight equations.
- **spaCy NER Extraction:** Entity extraction was replaced with specialized spaCy entity pipelines rather than pure regex.

### Reliability
- **Connection Pools & Retries:** `agent-brain` now handles database instability cleanly with exponential backoff rather than silent failures.
- **Circuit Breakers for Spread Activation:** Safely bounds explosion loops in dense memory hives via `max_activated_nodes` triggers.
- **`get_health()` probe:** Directly output Graph topology stats for deployment liveness probes.

## [0.1.0] - Initial Alpha
- Base feature set of associative graphs, entity persistence, and standalone `_default_embedder`.

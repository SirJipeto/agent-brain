# 🧠 Agent Brain — Associative Memory for AI Agents

> A Neo4j-powered associative memory system that gives AI agents brain-like, proactive recall.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Neo4j 5.x](https://img.shields.io/badge/Neo4j-5.x-008CC1.svg)](https://neo4j.com/)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)]()

---

## 🎯 What is Agent Brain?

Agent Brain is a **hybrid GraphRAG memory system** that combines semantic vector search with a living knowledge graph to give AI agents a persistent, associative memory — much like the human brain.

Unlike a plain vector store that only finds "similar" chunks, Agent Brain understands **relationships between concepts** and surfaces memories your agent didn't even know it needed.

| Technique | What it does |
|-----------|--------------|
| **Vector Search** | Semantic similarity across stored memories |
| **Graph Traversal** | Multi-hop reasoning through entity relationships |
| **Activation Spreading** | Brain-like ripple recall from a seed concept |
| **Proactive Memory** | Surfaces relevant context before the agent asks |
| **Memory Consolidation** | Auto-summarises stale conversations into facts |
| **Relationship Decay** | Prunes weak or forgotten connections over time |

### The "Aha" Moment

```python
# User: "I booked my flight to Japan!"
#
# Traditional RAG ──► Returns memories containing "Japan"
#
# Agent Brain     ──► Surfaces "Your passport expires in 8 months"
#                     because: passport → NEEDED_FOR → international_travel → INCLUDES → Japan
```

---

## 🚀 Features

- **Associative recall** — follows entity relationships up to *N* hops deep
- **Proactive brain loop** — every message triggers background retrieval; relevant memories are queued automatically
- **Natural context injection** — builds a `**Grounded Context:**` block ready to prepend to your LLM prompt
- **Cooldown & deduplication** — avoids surfacing the same memory twice in a short window
- **Memory consolidation** — groups old episodic memories by topic, condenses them into summary nodes, and archives the originals
- **Fact extraction** — pulls structured facts (dates, intentions, preferences) from conversation clusters
- **Relationship decay** — periodically weakens stale connections and prunes those below a minimum threshold
- **spaCy NER** — auto-classifies entities as people, organizations, places, dates, etc. (not just "concepts")
- **Pluggable embeddings** — choose between local (sentence-transformers), OpenAI API, or Ollama at install time
- **Framework agnostic** — works with Agent Zero, LangChain, Claude Code, or any custom agent

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- Neo4j 5.x (local or Aura cloud)

### Step 1 — Clone and install

```bash
git clone https://github.com/sirjipeto/agent-brain.git
cd agent-brain
```

### Step 2 — Choose your embedding provider

Agent Brain uses vector embeddings to power semantic search. You choose which embedding backend to use at install time:

| Install command | Provider | Runs where | Download size | Best for |
|----------------|----------|------------|---------------|----------|
| `pip install -e ".[local]"` | sentence-transformers | Your machine (CPU/GPU) | **~2 GB+** (includes PyTorch) | Privacy, offline use, no API costs |
| `pip install -e ".[openai]"` | OpenAI API | Cloud | ~1 MB | Best quality, lowest setup effort |
| `pip install -e ".[ollama]"` | Ollama | Your machine (via API) | ~1 MB (+ model pull) | Privacy + easy model management |

> ⚠️ **Note:** The `[local]` option downloads PyTorch and model weights (~2 GB+). If you're on a machine without a GPU or want faster setup, consider `[openai]` or `[ollama]`.

```bash
# Example: install with OpenAI embeddings
pip install -e ".[openai]"

# Example: install with local sentence-transformers
pip install -e ".[local]"

# Example: install all providers
pip install -e ".[all-embeddings]"
```

### Step 3 — Download the spaCy language model

Agent Brain uses spaCy for named entity recognition (extracting people, places, organizations, etc.):

```bash
python -m spacy download en_core_web_sm
```

### Step 4 — Run Neo4j with Docker (recommended)

```bash
docker run --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword \
  neo4j:5
```

### Step 5 — Configure your embedding provider

Set your provider via environment variables (or pass it in code):

```bash
# For sentence-transformers (default — no extra config needed)
export AGENT_BRAIN_EMBEDDING_PROVIDER=sentence-transformers

# For OpenAI
export AGENT_BRAIN_EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-your-key-here
# Optional: export AGENT_BRAIN_EMBEDDING_MODEL=text-embedding-3-small

# For Ollama (requires Ollama running locally)
export AGENT_BRAIN_EMBEDDING_PROVIDER=ollama
# Optional: export OLLAMA_BASE_URL=http://localhost:11434
# Pull the model first: ollama pull nomic-embed-text
```

Or configure in code:

```python
from agent_brain import Neo4jBrain, create_provider

# OpenAI
provider = create_provider("openai", api_key="sk-...")
brain = Neo4jBrain(embedder=provider)

# Ollama
provider = create_provider("ollama", model="nomic-embed-text")
brain = Neo4jBrain(embedder=provider)

# Local sentence-transformers
provider = create_provider("local", model="all-MiniLM-L6-v2")
brain = Neo4jBrain(embedder=provider)
```

> **No fallback.** If your chosen provider fails (missing dependency, bad API key, Ollama not running), Agent Brain will raise a clear error with install instructions — it will never silently fall back to a different model.

---

## 🔧 Quick Start

```python
from agent_brain import observe, get_context, respond

# 1. On every user message:
observe("I'm planning a trip to Japan next spring")

# 2. Before generating your response — grab proactive context:
context = get_context()
if context:
    print(context)
    # **Grounded Context:**
    # • Your passport expires in 8 months (related to Japan)
    # • You mentioned a budget of $3,000 (related to Japan)

# 3. After you generate a response, store it:
respond("Great! Spring is a wonderful time to visit for cherry blossoms.")
```

---

## 🚀 Integration Examples & Production Scaling

`agent-brain` is designed to be highly interoperable with popular agent frameworks and production architectures.

**Integration Guides:**
Check out the [`examples/`](./examples) directory for plug-and-play integrations:
- `fastapi_service.py`: Exposing agent-brain as a REST API with global connection pooling and health checks.
- `langchain_tool.py`: Wrapping hybrid search into a LangChain `BaseTool` for autonomous React agents.
- `claude_integration.py`: Direct Claude API integration showcasing how to explicitly map graph constraints into LLM context prompts.

**Production Deployment:**
If your graph spans millions of nodes, you should review [`SCALING.md`](./SCALING.md). `agent-brain` provides zero-overhead telemetry via `opentelemetry-api`. To query the health of your graph (useful for Kubernetes Liveness Probes), you can always call:
```python
health_stats = brain.get_health()
# {"status": "up", "metrics": {"total_memories": 1500, "total_entities": 400}}
```

---

## 🔌 Integration

Agent Brain is framework-agnostic — it works with any AI agent that processes messages. There are three ways to integrate it, from simplest to most flexible.

### 1. Convenience functions (simplest)

Drop these into your agent's message loop — no subclassing required:

```python
from agent_brain import observe, get_context, respond, recall

# On every user message:
observe("I'm planning a trip to Japan", role="user")

# Before generating your response — grab proactive context:
context = get_context()          # returns a "**Grounded Context:**" block or ""

# After generating a response, store it:
respond("Spring is a wonderful time for cherry blossoms.")

# On-demand recall (e.g. as a tool the agent can call):
results = recall("Japan travel budget", limit=5)
```

### 2. BrainAgent wrapper

Wraps the full lifecycle into a single object you can embed in any agent class:

```python
from agent_brain import BrainAgent

class MyAgent:
    def __init__(self):
        self.brain = BrainAgent()
        self.brain.initialize()

    def think(self, user_input: str) -> str:
        self.brain.observe(user_input)
        context = self.brain.get_context()
        response = self.llm.generate(f"{context}\n{user_input}")
        self.brain.respond(response)
        return response
```

### 3. ObserverFramework (full control)

For conversation scoping, hook callbacks, and rich manual recall:

```python
from agent_brain import Neo4jBrain
from agent_brain.observer import ObserverFramework

brain = Neo4jBrain()
observer = ObserverFramework(brain)

# Register a hook that fires on every stored message
observer.register_hook(lambda event_type, data: print(f"[{event_type}] {data.content[:60]}"))

observer.start_conversation(topic="Trip planning")
observer.on_message("I want to visit Kyoto in April")

context = observer.get_grounded_context()   # inject into your prompt

# Spontaneous manual recall
result = observer.recall_related("Japan travel budget")
# {
#   "semantic_matches": [...],
#   "associations": [...],
#   "activated_concepts": [("passport", 0.87), ("visa", 0.73), ...]
# }

observer.end_conversation()   # auto-summarises recent exchanges
```

---

## 🏭 Integrating into an Agent Framework

If you're building or extending an agent framework (like [OpenClaw](https://openclaw.ai)), here's the recommended architecture for wiring Agent Brain into the agent loop. This pattern applies to any framework — OpenClaw is used as a reference example.

### Step 1 — Install Agent Brain as a dependency

```bash
# Inside your framework's virtual environment (pick your embedding backend):
pip install "agent-brain[openai] @ git+https://github.com/sirjipeto/agent-brain.git"
# or: pip install "agent-brain[local] @ git+https://github.com/sirjipeto/agent-brain.git"

# Download the spaCy NER model
python -m spacy download en_core_web_sm
```

Set connection and embedding variables (or `.env`):

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=yourpassword
export AGENT_BRAIN_EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-your-key-here
```

### Step 2 — Hook into the message lifecycle

Every agent framework has a message loop. You need to tap into three points:

```
 User message ──► [1] OBSERVE ──► Agent LLM ──► [2] INJECT CONTEXT ──► Response ──► [3] STORE
```

```python
# brain_middleware.py — drop into your framework's middleware / plugin directory
from agent_brain import BrainAgent

brain = BrainAgent()
brain.initialize()

# [1] Called when a user message arrives
def before_llm(user_message: str) -> str:
    """Observe the message and return enriched prompt."""
    brain.observe(user_message, role="user")
    context = brain.get_context()
    if context:
        return f"{context}\n\n{user_message}"
    return user_message

# [2] Called after the LLM generates a response
def after_llm(response: str):
    """Store the agent's response as a memory."""
    brain.respond(response)
```

### Step 3 — Expose recall as a tool

Let the agent call `recall` on demand so it can search its own memory:

```python
from agent_brain import recall

# Register this as a tool your agent can invoke
def remember(query: str) -> str:
    """Search the agent's associative memory for relevant context."""
    results = recall(query, limit=5)
    if not results:
        return "No relevant memories found."
    return "\n".join(f"• {r['memory']}" for r in results)
```

Register it in your framework's tool config — for example in OpenClaw:

```json
{
  "mcpServers": {},
  "customTools": [
    {
      "name": "remember",
      "description": "Search associative memory for past context relevant to the current task",
      "module": "brain_middleware",
      "function": "remember"
    }
  ]
}
```

### Step 4 — Schedule maintenance

Run consolidation and decay on a timer (background thread, cron job, or framework scheduler):

```python
from agent_brain import consolidate_memories, decay_connections

# e.g. nightly or every N conversations
def run_maintenance():
    consolidate_memories(older_than_days=7)
    decay_connections(threshold=0.1)
```

### Architecture summary

```
┌──────────────────────────────────────────────────────┐
│                YOUR AGENT FRAMEWORK                   │
│                                                       │
│   User msg ──► before_llm() ──► LLM ──► after_llm() │
│                     │                       │         │
│                     ▼                       ▼         │
│              ┌────────────────────────────────┐       │
│              │         AGENT BRAIN            │       │
│              │                                │       │
│              │  observe() → extract entities  │       │
│              │  get_context() → graph search  │       │
│              │  respond() → store response    │       │
│              │  recall() → associative search │       │
│              │                                │       │
│              │         Neo4j Graph            │       │
│              └────────────────────────────────┘       │
│                                                       │
│   Tools: [ remember() ]     Scheduler: maintenance()  │
└──────────────────────────────────────────────────────┘
```

---

## 📚 API Reference

### Convenience functions (`from agent_brain import ...`)

| Function | Signature | Description |
|----------|-----------|-------------|
| `observe` | `(message, role="user") → memory_id` | Store a message and trigger proactive retrieval |
| `respond` | `(response) → memory_id` | Store the agent's reply |
| `get_context` | `() → str` | Return the current **Grounded Context** block |
| `recall` | `(topic, limit=5) → List[Dict]` | Entity-based associative recall |
| `search` | `(query) → List[Dict]` | Keyword/full-text search over stored memories |
| `get_stats` | `() → Dict` | Graph statistics: memory count, entity count, relationship count |

### Embedding providers (`from agent_brain import ...`)

| Class / Function | Description |
|-----------------|-------------|
| `create_provider(name, **kwargs)` | Factory: create a provider by name ("local", "openai", "ollama") |
| `create_provider_from_env()` | Create a provider from `AGENT_BRAIN_EMBEDDING_*` env vars |
| `list_providers()` | List available providers with install commands |
| `SentenceTransformerProvider` | Local embeddings (384d default, `all-MiniLM-L6-v2`) |
| `OpenAIProvider` | OpenAI API embeddings (1536d default, `text-embedding-3-small`) |
| `OllamaProvider` | Ollama local API embeddings (768d default, `nomic-embed-text`) |
| `CallableProvider` | Wrap any `fn(text) → List[float]` as a provider (testing/custom) |

### Memory maintenance

| Function | Signature | Description |
|----------|-----------|-------------|
| `consolidate_memories` | `(older_than_days=7, dry_run=False) → Dict` | Summarise and archive stale memories |
| `decay_connections` | `(threshold=0.1, dry_run=False) → Dict` | Weaken and prune low-weight relationships |
| `get_maintenance_status` | `() → Dict` | Report on old memories, archived nodes, and weak connections |

### BrainAgent methods

| Method | Description |
|--------|-------------|
| `initialize()` | Connect to Neo4j and set up schema |
| `start_conversation()` | Reset in-memory context window |
| `observe(message, role)` | Store message + update grounded context |
| `respond(response)` | Store agent response |
| `get_context()` | Return current grounded context string |
| `recall(topic, limit)` | Associative recall by entity |
| `search(query)` | Full-text keyword search |
| `get_stats()` | Graph statistics |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      YOUR AGENT                          │
│    observe(msg)  ──►  get_context()  ──►  respond()     │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     AGENT BRAIN                          │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │            OBSERVER FRAMEWORK                    │    │
│  │  • Entity extraction from each message          │    │
│  │  • Graph traversal (multi-hop, depth=3)         │    │
│  │  • Relevance evaluation (6-factor scoring)      │    │
│  │  • Proactive memory queue                       │    │
│  │  • Cooldown / deduplication                     │    │
│  │  • Hook callbacks for custom side-effects       │    │
│  └───────────────────────┬─────────────────────────┘    │
│                          │                               │
│            ┌─────────────┼──────────────┐               │
│            ▼             ▼              ▼                │
│  ┌──────────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ CONSOLIDATOR │  │  NEO4J   │  │ RELATIONSHIP     │   │
│  │ (summariser) │  │  GRAPH   │  │ DECAY ENGINE     │   │
│  │ • Clustering │  │ Memory   │  │ • weight * 0.95  │   │
│  │ • Fact ext.  │  │ Entity   │  │ • prune < 0.1    │   │
│  │ • Archiving  │  │ Fact     │  │ • age-based      │   │
│  └──────────────┘  │ Concept  │  └──────────────────┘   │
│                    │ + Vector │                          │
│                    └──────────┘                          │
└─────────────────────────────────────────────────────────┘
```

### Graph Schema

| Node label | Key properties |
|------------|---------------|
| `Memory` | `id`, `content`, `summary`, `content_type`, `importance`, `archived`, `created_at` |
| `Entity` | `name` |
| `Fact` | `subject`, `predicate`, `value`, `confidence` |

| Relationship | Meaning |
|-------------|---------|
| `MENTIONS` | Memory → Entity |
| `RELATED_TO` | Entity ↔ Entity (weighted) |
| `HAS_FACT` | Entity → Fact |
| `ABOUT` | Fact → Entity |

---

## 🔒 Environment Variables

### Neo4j Connection

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |

### Embedding Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_BRAIN_EMBEDDING_PROVIDER` | `sentence-transformers` | Provider name: `sentence-transformers`, `openai`, or `ollama` |
| `AGENT_BRAIN_EMBEDDING_MODEL` | *(provider default)* | Model name (e.g. `all-MiniLM-L6-v2`, `text-embedding-3-small`, `nomic-embed-text`) |
| `OPENAI_API_KEY` | — | Required for the `openai` provider |
| `OPENAI_BASE_URL` | — | Custom OpenAI-compatible API endpoint |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=mysecretpassword
export AGENT_BRAIN_EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-your-key-here
```

---

## 🛠️ Memory Maintenance

Agent Brain ships with built-in hygiene utilities to keep the graph lean:

```python
from agent_brain import consolidate_memories, decay_connections, get_maintenance_status

# Check what needs attention
status = get_maintenance_status()
# {'old_memories': 42, 'archived_memories': 108, 'weak_connections': 17,
#  'consolidation_needed': True}

# Preview what consolidation would do (dry run)
consolidate_memories(older_than_days=7, dry_run=True)

# Run it for real
result = consolidate_memories(older_than_days=7)
print(result)
# {'memories_processed': 42, 'memories_archived': 38,
#  'memories_consolidated': 5, 'facts_extracted': 14, ...}

# Decay and prune weak relationships
decay_connections(threshold=0.1)
```

Run these on a schedule (e.g., nightly cron or a background thread) to prevent unbounded graph growth.

---

## 🤝 Contributing

Contributions welcome! Please open an issue first to discuss significant changes, then submit a PR.

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built on [Neo4j](https://neo4j.com/) for graph storage and native vector search
- Inspired by [GraphRAG](https://arxiv.org/abs/2404.16130) research from Microsoft
- Activation spreading model inspired by human associative memory research

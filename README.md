# 🧠 Agent Brain — Associative Memory for AI Agents

> A Neo4j-powered associative memory system that gives AI agents brain-like, proactive recall.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Neo4j 5.x](https://img.shields.io/badge/Neo4j-5.x-008CC1.svg)](https://neo4j.com/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)]()

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
- **Framework agnostic** — works with Agent Zero, LangChain, Claude Code, or any custom agent

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- Neo4j 5.x (local or Aura cloud)

### Install from source

```bash
git clone https://github.com/sirjipeto/agent-brain.git
cd agent-brain
pip install -e .                          # core (neo4j driver only)
pip install -e ".[embeddings]"            # + sentence-transformers for vector search
```

### Run Neo4j with Docker (recommended)

```bash
docker run --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword \
  neo4j:5
```

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
# Inside your framework's virtual environment:
pip install git+https://github.com/sirjipeto/agent-brain.git
# Or for vector search support:
pip install "agent-brain[embeddings] @ git+https://github.com/sirjipeto/agent-brain.git"
```

Set Neo4j connection via environment variables (or `.env`):

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=yourpassword
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

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |

Override them before importing the library:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=mysecretpassword
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

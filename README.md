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

## 🔌 Framework Integration

Agent Brain is framework-agnostic — it works with any AI agent that processes messages. Below are setup guides for the most popular frameworks.

### Agent Zero

```python
from agent_brain import observe, get_context

def on_user_message(message: str) -> str:
    observe(message, role="user")
    context = get_context()
    return f"System: {context}\n\nUser: {message}" if context else message
```

---

### OpenClaw

#### 1. Install OpenClaw

```bash
# macOS / Linux / WSL2 (one-line installer)
curl -fsSL https://docs.openclaw.ai/install.sh | bash

# Or via npm
npm install -g openclaw
```

> **Prerequisites:** Node.js 22+, Git, and an API key from a supported provider (Anthropic, OpenAI, Google, etc.)

#### 2. Install Agent Brain into OpenClaw's environment

```bash
# From your agent-brain directory:
pip install -e ".[embeddings]"
```

#### 3. Set environment variables

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=yourpassword
```

#### 4. Wire into OpenClaw's tool / hook system

```python
# openclaw_brain_tool.py — register as a custom tool in OpenClaw
from agent_brain import observe, get_context, recall

def on_user_message(message: str) -> str:
    """Hook into OpenClaw's message pipeline."""
    observe(message, role="user")
    context = get_context()
    return f"System: {context}\n\nUser: {message}" if context else message

def remember(query: str) -> str:
    """Expose as an OpenClaw tool so the agent can recall memories on demand."""
    results = recall(query, limit=5)
    return "\n".join(r["memory"] for r in results) if results else "No memories found."
```

Register the tool in your OpenClaw config (`.openclaw/config.json` or via the CLI):

```json
{
  "tools": [
    {
      "name": "remember",
      "description": "Search associative memory for relevant past context",
      "module": "openclaw_brain_tool",
      "function": "remember"
    }
  ]
}
```

---

### Hermes Agent (Nous Research)

#### 1. Install Hermes Agent

```bash
# macOS / Linux / WSL2 (one-line installer — installs uv, Python 3.11, Node, ripgrep, ffmpeg)
curl -fsSL https://hermes.nousresearch.com/install.sh | bash

# Or manually via Git + uv
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
uv sync
```

> **Prerequisites:** Git (installer handles the rest). Windows users must use WSL2.

#### 2. Install Agent Brain into Hermes' virtual environment

```bash
# Activate the Hermes venv first, then:
pip install -e /path/to/agent-brain[embeddings]
```

#### 3. Set environment variables

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=yourpassword
```

#### 4. Wire into Hermes' agent loop

```python
# hermes_brain_plugin.py — add to Hermes' plugins directory
from agent_brain import observe, get_context, recall, respond

class BrainPlugin:
    """Hermes Agent plugin that adds associative memory."""

    name = "agent_brain"

    def on_user_message(self, message: str) -> str:
        """Called on every user message. Returns context to prepend to prompt."""
        observe(message, role="user")
        return get_context()

    def on_agent_response(self, response: str):
        """Called after the agent generates a response."""
        respond(response)

    def recall_memory(self, query: str, limit: int = 5) -> str:
        """Expose as a callable tool for the agent."""
        results = recall(query, limit=limit)
        return "\n".join(r["memory"] for r in results) if results else "No memories found."
```

Register in your Hermes config:

```yaml
# hermes.yaml
plugins:
  - module: hermes_brain_plugin
    class: BrainPlugin
```

---

### Claude Code / Custom tool

```python
from agent_brain import recall

def remember(query: str) -> str:
    results = recall(query, limit=5)
    return "\n".join(r["memory"] for r in results)
```

### Full BrainAgent wrapper (any framework)

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

### Using the ObserverFramework directly

The `ObserverFramework` gives you finer control: conversation scoping, hook callbacks, and rich manual recall.

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

"""
Agent Brain - Associative Memory for AI Agents

A Neo4j-powered brain-like memory system for AI agents.

Usage:
    from agent_brain import BrainAgent, observe, get_context
    
    brain = BrainAgent()
    brain.initialize()
    
    observe("User message")
    context = get_context()
"""

from .brain import Neo4jBrain
from .observer import ObserverFramework
from .embeddings import (
    EmbeddingProvider,
    SentenceTransformerProvider,
    OpenAIProvider,
    OllamaProvider,
    CallableProvider,
    create_provider,
    create_provider_from_env,
    list_providers,
    EmbeddingError,
    ProviderNotAvailableError,
)
from .agent_integration import (
    BrainAgent, observe, respond, get_context, recall, search, get_stats,
    consolidate_memories, decay_connections, get_maintenance_status,
)

__all__ = [
    "Neo4jBrain",
    "ObserverFramework",
    # Embedding providers
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "CallableProvider",
    "create_provider",
    "create_provider_from_env",
    "list_providers",
    "EmbeddingError",
    "ProviderNotAvailableError",
    # Agent integration
    "BrainAgent",
    "observe",
    "respond",
    "get_context",
    "recall",
    "search",
    "get_stats",
    "consolidate_memories",
    "decay_connections",
    "get_maintenance_status",
]


__version__ = "0.1.0"

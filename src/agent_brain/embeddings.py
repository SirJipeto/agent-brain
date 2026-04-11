"""
Pluggable Embedding Providers for agent_brain.

Provides an EmbeddingProvider protocol and concrete implementations for
multiple embedding backends. Users choose their provider at install/setup time.

No fallback chain — if the configured provider fails, it fails loudly with
a clear error message telling the user how to fix it.

Supported providers:
- SentenceTransformerProvider: Local embeddings via sentence-transformers
- OpenAIProvider: API-based embeddings via OpenAI
- OllamaProvider: Local API embeddings via Ollama

Usage:
    # Option 1: Direct instantiation
    from agent_brain.embeddings import SentenceTransformerProvider
    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    brain = Neo4jBrain(connection=conn, embedder=provider)

    # Option 2: Factory from config
    from agent_brain.embeddings import create_provider
    provider = create_provider("openai", model="text-embedding-3-small", api_key="sk-...")
    brain = Neo4jBrain(connection=conn, embedder=provider)

    # Option 3: From environment
    from agent_brain.embeddings import create_provider_from_env
    provider = create_provider_from_env()  # reads AGENT_BRAIN_EMBEDDING_PROVIDER, etc.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, runtime_checkable, Protocol

logger = logging.getLogger(__name__)


# ============================================================================
# EMBEDDING PROVIDER PROTOCOL
# ============================================================================

@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol for embedding providers.

    Any object with an `embed(text) -> List[float]` method and a `dimension`
    property satisfies this protocol. This means raw callables do NOT satisfy
    the protocol — use `CallableProvider` to wrap them if needed.
    """

    @property
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        ...

    @property
    def provider_name(self) -> str:
        """Human-readable provider name for logging and diagnostics."""
        ...

    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.

        Args:
            text: Input text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            EmbeddingError: If the embedding fails for any reason.
        """
        ...


# ============================================================================
# EXCEPTIONS
# ============================================================================

class EmbeddingError(Exception):
    """Raised when an embedding operation fails."""

    def __init__(self, message: str, provider: str = "unknown", cause: Optional[Exception] = None):
        self.provider = provider
        self.cause = cause
        super().__init__(f"[{provider}] {message}")


class ProviderNotAvailableError(EmbeddingError):
    """Raised when a provider's dependencies are not installed."""

    def __init__(self, provider: str, install_hint: str, cause: Optional[Exception] = None):
        self.install_hint = install_hint
        super().__init__(
            f"Provider '{provider}' is not available. Install with: {install_hint}",
            provider=provider,
            cause=cause,
        )


# ============================================================================
# BASE CLASS
# ============================================================================

class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    Provides common validation and error handling.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._initialized = False

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""
        ...

    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize the provider (load model, validate API key, etc.).
        Called on first use. Should raise ProviderNotAvailableError if
        dependencies are missing.
        """
        ...

    @abstractmethod
    def _embed_impl(self, text: str) -> List[float]:
        """
        Internal embedding implementation. Override this in subclasses.
        """
        ...

    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        Initializes the provider on first call.
        Raises EmbeddingError on failure — NO silent fallback.
        """
        if not self._initialized:
            try:
                self._initialize()
                self._initialized = True
                logger.info(
                    f"Embedding provider '{self.provider_name}' initialized "
                    f"(model={self._model_name}, dim={self.dimension})"
                )
            except ProviderNotAvailableError:
                raise  # Re-raise directly
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to initialize: {e}",
                    provider=self.provider_name,
                    cause=e,
                )

        try:
            vector = self._embed_impl(text)
        except EmbeddingError:
            raise  # Already wrapped
        except Exception as e:
            raise EmbeddingError(
                f"Embedding failed for text ({len(text)} chars): {e}",
                provider=self.provider_name,
                cause=e,
            )

        # Validate output dimension
        if len(vector) != self.dimension:
            logger.warning(
                f"Expected {self.dimension}-dim vector from {self.provider_name}, "
                f"got {len(vector)}-dim. This may cause vector index errors."
            )

        return vector

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self._model_name!r}, dim={self.dimension})"


# ============================================================================
# SENTENCE-TRANSFORMERS PROVIDER (LOCAL)
# ============================================================================

class SentenceTransformerProvider(BaseEmbeddingProvider):
    """
    Local embeddings via sentence-transformers.

    Install: pip install agent-brain[local]
    Models: all-MiniLM-L6-v2 (384d), all-mpnet-base-v2 (768d), etc.
    """

    # Known model dimensions
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "multi-qa-MiniLM-L6-cos-v1": 384,
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self._model = None
        self._dim = self.MODEL_DIMENSIONS.get(model_name, 384)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return "sentence-transformers"

    def _initialize(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ProviderNotAvailableError(
                provider="sentence-transformers",
                install_hint="pip install agent-brain[local]  (or: pip install sentence-transformers)",
                cause=e,
            )

        logger.info(f"Loading sentence-transformers model '{self._model_name}'...")
        self._model = SentenceTransformer(self._model_name)

        # Get actual dimension from model
        test_vec = self._model.encode("test")
        self._dim = len(test_vec)

    def _embed_impl(self, text: str) -> List[float]:
        embedding = self._model.encode(text)
        return embedding.tolist()


# ============================================================================
# OPENAI PROVIDER (API)
# ============================================================================

class OpenAIProvider(BaseEmbeddingProvider):
    """
    API-based embeddings via OpenAI.

    Install: pip install agent-brain[openai]
    Models: text-embedding-3-small (1536d), text-embedding-3-large (3072d),
            text-embedding-ada-002 (1536d)
    """

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = "text-embedding-3-small",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        super().__init__(model_name)
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url
        self._client = None
        self._dim = self.MODEL_DIMENSIONS.get(model_name, 1536)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return "openai"

    def _initialize(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderNotAvailableError(
                provider="openai",
                install_hint="pip install agent-brain[openai]  (or: pip install openai)",
                cause=e,
            )

        if not self._api_key:
            raise EmbeddingError(
                "API key required. Set OPENAI_API_KEY env var or pass api_key= parameter.",
                provider="openai",
            )

        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url

        self._client = OpenAI(**kwargs)

    def _embed_impl(self, text: str) -> List[float]:
        response = self._client.embeddings.create(
            input=text,
            model=self._model_name,
        )
        return response.data[0].embedding


# ============================================================================
# OLLAMA PROVIDER (LOCAL API)
# ============================================================================

class OllamaProvider(BaseEmbeddingProvider):
    """
    Local API embeddings via Ollama.

    Install: pip install agent-brain[ollama]
    Requires: Ollama running locally (default http://localhost:11434)
    Models: nomic-embed-text (768d), mxbai-embed-large (1024d), all-minilm (384d)
    """

    MODEL_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
    }

    def __init__(self, model_name: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self._base_url = base_url
        self._dim = self.MODEL_DIMENSIONS.get(model_name, 768)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _initialize(self) -> None:
        try:
            import httpx
        except ImportError as e:
            raise ProviderNotAvailableError(
                provider="ollama",
                install_hint="pip install agent-brain[ollama]  (or: pip install httpx)",
                cause=e,
            )

        # Verify Ollama is running
        try:
            client = httpx.Client(timeout=5.0)
            resp = client.get(f"{self._base_url}/api/version")
            resp.raise_for_status()
            client.close()
        except Exception as e:
            raise EmbeddingError(
                f"Cannot connect to Ollama at {self._base_url}. "
                f"Ensure Ollama is running: https://ollama.com/download",
                provider="ollama",
                cause=e,
            )

    def _embed_impl(self, text: str) -> List[float]:
        import httpx

        response = httpx.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model_name, "prompt": text},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        embedding = data.get("embedding", [])
        if not embedding:
            raise EmbeddingError(
                f"Empty embedding returned for model '{self._model_name}'. "
                f"Ensure the model is pulled: ollama pull {self._model_name}",
                provider="ollama",
            )

        self._dim = len(embedding)  # Auto-detect dimension
        return embedding


# ============================================================================
# CALLABLE WRAPPER (for backward compatibility & testing)
# ============================================================================

class CallableProvider(BaseEmbeddingProvider):
    """
    Wraps a raw callable (text -> List[float]) as an EmbeddingProvider.
    Used for backward compatibility and testing.
    """

    def __init__(self, fn, dimension: int = 384, name: str = "callable"):
        super().__init__(name)
        self._fn = fn
        self._dim = dimension
        self._name = name

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return self._name

    def _initialize(self) -> None:
        pass  # Nothing to initialize

    def _embed_impl(self, text: str) -> List[float]:
        return self._fn(text)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

# Provider registry
PROVIDERS = {
    "sentence-transformers": SentenceTransformerProvider,
    "local": SentenceTransformerProvider,  # alias
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


def create_provider(provider_name: str, **kwargs) -> BaseEmbeddingProvider:
    """
    Create an embedding provider by name.

    Args:
        provider_name: One of "sentence-transformers"/"local", "openai", "ollama"
        **kwargs: Provider-specific configuration (model_name, api_key, etc.)

    Returns:
        Configured EmbeddingProvider instance.

    Raises:
        ValueError: If provider_name is unknown.
    """
    provider_name = provider_name.lower().strip()
    cls = PROVIDERS.get(provider_name)

    if cls is None:
        available = ", ".join(sorted(set(k for k, v in PROVIDERS.items())))
        raise ValueError(
            f"Unknown embedding provider: '{provider_name}'. "
            f"Available: {available}"
        )

    # Map generic 'model' kwarg to 'model_name' if needed
    if "model" in kwargs and "model_name" not in kwargs:
        kwargs["model_name"] = kwargs.pop("model")

    return cls(**kwargs)


def create_provider_from_env() -> BaseEmbeddingProvider:
    """
    Create an embedding provider from environment variables.

    Environment variables:
        AGENT_BRAIN_EMBEDDING_PROVIDER: Provider name (default: "sentence-transformers")
        AGENT_BRAIN_EMBEDDING_MODEL: Model name (provider-specific default)
        OPENAI_API_KEY: Required for OpenAI provider
        OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)

    Returns:
        Configured EmbeddingProvider instance.
    """
    provider_name = os.getenv("AGENT_BRAIN_EMBEDDING_PROVIDER", "sentence-transformers")

    kwargs: Dict[str, Any] = {}

    model = os.getenv("AGENT_BRAIN_EMBEDDING_MODEL")
    if model:
        kwargs["model_name"] = model

    # Provider-specific env vars
    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url

    elif provider_name == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url

    return create_provider(provider_name, **kwargs)


def list_providers() -> Dict[str, str]:
    """List available providers with their install commands."""
    return {
        "sentence-transformers": "pip install agent-brain[local]",
        "openai": "pip install agent-brain[openai]",
        "ollama": "pip install agent-brain[ollama]",
    }

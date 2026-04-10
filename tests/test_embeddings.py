"""
Tests for the pluggable embedding system.

Covers:
- EmbeddingProvider protocol
- CallableProvider (backward compat, testing)
- SentenceTransformerProvider (init, missing deps)
- OpenAIProvider (init, missing API key)
- OllamaProvider (init, missing server)
- Factory functions (create_provider, create_provider_from_env)
- Error handling (no fallback, loud failures)
- Brain integration (provider passed to Neo4jBrain)
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from agent_brain.embeddings import (
    EmbeddingProvider,
    BaseEmbeddingProvider,
    CallableProvider,
    SentenceTransformerProvider,
    OpenAIProvider,
    OllamaProvider,
    EmbeddingError,
    ProviderNotAvailableError,
    create_provider,
    create_provider_from_env,
    list_providers,
    PROVIDERS,
)
from agent_brain.brain import Neo4jBrain
from tests.conftest import FakeNeo4jConnection, mock_embedder


# ============================================================================
# CALLABLE PROVIDER
# ============================================================================

class TestCallableProvider:
    """Test that raw callables are properly wrapped."""

    def test_wraps_callable(self):
        def my_fn(text):
            return [0.0] * 384
        provider = CallableProvider(my_fn)
        result = provider.embed("test")
        assert len(result) == 384

    def test_respects_dimension(self):
        provider = CallableProvider(lambda t: [0.0] * 768, dimension=768)
        assert provider.dimension == 768

    def test_returns_provider_name(self):
        provider = CallableProvider(lambda t: [], name="test-embedder")
        assert provider.provider_name == "test-embedder"

    def test_wraps_mock_embedder(self):
        provider = CallableProvider(mock_embedder, dimension=384, name="mock")
        result = provider.embed("hello world")
        assert len(result) == 384

    def test_deterministic_output(self):
        provider = CallableProvider(mock_embedder)
        a = provider.embed("test")
        b = provider.embed("test")
        assert a == b, "Same input should produce same output"

    def test_raises_embedding_error_on_failure(self):
        def failing_fn(text):
            raise RuntimeError("GPU out of memory")

        provider = CallableProvider(failing_fn)
        with pytest.raises(EmbeddingError) as exc_info:
            provider.embed("test")
        assert "Embedding failed" in str(exc_info.value)
        assert exc_info.value.cause is not None


# ============================================================================
# SENTENCE TRANSFORMER PROVIDER
# ============================================================================

class TestSentenceTransformerProvider:

    def test_has_correct_default_model(self):
        provider = SentenceTransformerProvider()
        assert provider._model_name == "all-MiniLM-L6-v2"

    def test_has_correct_default_dimension(self):
        provider = SentenceTransformerProvider()
        assert provider.dimension == 384

    def test_known_model_dimensions(self):
        provider768 = SentenceTransformerProvider("all-mpnet-base-v2")
        assert provider768.dimension == 768

    def test_provider_name(self):
        provider = SentenceTransformerProvider()
        assert provider.provider_name == "sentence-transformers"

    def test_repr(self):
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        assert "all-MiniLM-L6-v2" in repr(provider)

    def test_raises_if_not_installed(self):
        """Should raise ProviderNotAvailableError with install hint."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            provider = SentenceTransformerProvider()
            # Force re-init by resetting state
            provider._initialized = False
            with pytest.raises(ProviderNotAvailableError) as exc_info:
                provider.embed("test")
            assert "pip install" in str(exc_info.value)
            assert "sentence-transformers" in exc_info.value.install_hint


# ============================================================================
# OPENAI PROVIDER
# ============================================================================

class TestOpenAIProvider:

    def test_has_correct_default_model(self):
        provider = OpenAIProvider()
        assert provider._model_name == "text-embedding-3-small"

    def test_has_correct_default_dimension(self):
        provider = OpenAIProvider()
        assert provider.dimension == 1536

    def test_known_model_dimensions(self):
        large = OpenAIProvider(model_name="text-embedding-3-large")
        assert large.dimension == 3072

    def test_provider_name(self):
        assert OpenAIProvider().provider_name == "openai"

    def test_reads_api_key_from_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            provider = OpenAIProvider()
            assert provider._api_key == "sk-test123"

    def test_explicit_api_key_overrides_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}):
            provider = OpenAIProvider(api_key="sk-explicit")
            assert provider._api_key == "sk-explicit"

    def test_raises_if_no_api_key_or_not_installed(self):
        """Should raise EmbeddingError about missing API key, or ProviderNotAvailableError if openai not installed."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            provider = OpenAIProvider(api_key=None)
            provider._api_key = None  # Ensure no key
            with pytest.raises(EmbeddingError) as exc_info:
                provider.embed("test")
            # Either "API key required" (openai installed) or "not available" (not installed)
            msg = str(exc_info.value)
            assert "API key required" in msg or "not available" in msg

    def test_raises_if_not_installed(self):
        with patch.dict("sys.modules", {"openai": None}):
            provider = OpenAIProvider(api_key="sk-test")
            provider._initialized = False
            with pytest.raises(ProviderNotAvailableError) as exc_info:
                provider.embed("test")
            assert "openai" in exc_info.value.install_hint


# ============================================================================
# OLLAMA PROVIDER
# ============================================================================

class TestOllamaProvider:

    def test_has_correct_default_model(self):
        provider = OllamaProvider()
        assert provider._model_name == "nomic-embed-text"

    def test_has_correct_default_dimension(self):
        assert OllamaProvider().dimension == 768

    def test_provider_name(self):
        assert OllamaProvider().provider_name == "ollama"

    def test_custom_base_url(self):
        provider = OllamaProvider(base_url="http://my-server:11434")
        assert provider._base_url == "http://my-server:11434"

    def test_raises_if_httpx_not_installed(self):
        with patch.dict("sys.modules", {"httpx": None}):
            provider = OllamaProvider()
            provider._initialized = False
            with pytest.raises(ProviderNotAvailableError) as exc_info:
                provider.embed("test")
            assert "httpx" in exc_info.value.install_hint


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

class TestCreateProvider:

    def test_creates_sentence_transformers(self):
        provider = create_provider("sentence-transformers")
        assert isinstance(provider, SentenceTransformerProvider)

    def test_local_alias(self):
        provider = create_provider("local")
        assert isinstance(provider, SentenceTransformerProvider)

    def test_creates_openai(self):
        provider = create_provider("openai", api_key="sk-test")
        assert isinstance(provider, OpenAIProvider)

    def test_creates_ollama(self):
        provider = create_provider("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_case_insensitive(self):
        provider = create_provider("OPENAI", api_key="sk-test")
        assert isinstance(provider, OpenAIProvider)

    def test_strips_whitespace(self):
        provider = create_provider("  local  ")
        assert isinstance(provider, SentenceTransformerProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError) as exc_info:
            create_provider("invalid-provider")
        assert "Unknown embedding provider" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_model_kwarg_mapped(self):
        provider = create_provider("local", model="all-mpnet-base-v2")
        assert provider._model_name == "all-mpnet-base-v2"

    def test_model_name_kwarg(self):
        provider = create_provider("local", model_name="all-MiniLM-L12-v2")
        assert provider._model_name == "all-MiniLM-L12-v2"


class TestCreateProviderFromEnv:

    def test_defaults_to_sentence_transformers(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AGENT_BRAIN_EMBEDDING_PROVIDER", None)
            os.environ.pop("AGENT_BRAIN_EMBEDDING_MODEL", None)
            provider = create_provider_from_env()
            assert isinstance(provider, SentenceTransformerProvider)

    def test_reads_provider_from_env(self):
        with patch.dict(os.environ, {"AGENT_BRAIN_EMBEDDING_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}):
            provider = create_provider_from_env()
            assert isinstance(provider, OpenAIProvider)

    def test_reads_model_from_env(self):
        with patch.dict(os.environ, {
            "AGENT_BRAIN_EMBEDDING_PROVIDER": "local",
            "AGENT_BRAIN_EMBEDDING_MODEL": "all-mpnet-base-v2"
        }):
            provider = create_provider_from_env()
            assert provider._model_name == "all-mpnet-base-v2"

    def test_openai_reads_api_key(self):
        with patch.dict(os.environ, {
            "AGENT_BRAIN_EMBEDDING_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-from-env",
        }):
            provider = create_provider_from_env()
            assert isinstance(provider, OpenAIProvider)
            assert provider._api_key == "sk-from-env"

    def test_ollama_reads_base_url(self):
        with patch.dict(os.environ, {
            "AGENT_BRAIN_EMBEDDING_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://gpu-box:11434",
        }):
            provider = create_provider_from_env()
            assert isinstance(provider, OllamaProvider)
            assert provider._base_url == "http://gpu-box:11434"


class TestListProviders:
    def test_returns_all_providers(self):
        providers = list_providers()
        assert "sentence-transformers" in providers
        assert "openai" in providers
        assert "ollama" in providers

    def test_includes_install_commands(self):
        providers = list_providers()
        for name, cmd in providers.items():
            assert "pip install" in cmd


# ============================================================================
# ERROR SEMANTICS
# ============================================================================

class TestErrorSemantics:
    """Verify that providers fail loudly — no silent fallback."""

    def test_embedding_error_has_provider(self):
        err = EmbeddingError("test error", provider="openai")
        assert err.provider == "openai"
        assert "[openai]" in str(err)

    def test_provider_not_available_has_install_hint(self):
        err = ProviderNotAvailableError(
            provider="sentence-transformers",
            install_hint="pip install sentence-transformers"
        )
        assert err.install_hint == "pip install sentence-transformers"
        assert "not available" in str(err)

    def test_no_silent_fallback_on_init_failure(self):
        """Providers must NOT fall back to hash-based mock on failure."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            provider = SentenceTransformerProvider()
            provider._initialized = False
            with pytest.raises(ProviderNotAvailableError):
                provider.embed("this should not produce a hash vector")


# ============================================================================
# BRAIN INTEGRATION
# ============================================================================

class TestBrainEmbeddingIntegration:
    """Test that Neo4jBrain works with pluggable providers."""

    def test_accepts_callable(self):
        """Backward compat: raw callable wraps in CallableProvider."""
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)
        # Should have wrapped in CallableProvider
        assert isinstance(brain._provider, CallableProvider)

    def test_accepts_provider_instance(self):
        conn = FakeNeo4jConnection()
        provider = CallableProvider(mock_embedder, name="test")
        brain = Neo4jBrain(connection=conn, embedder=provider)
        assert brain._provider is provider

    def test_embed_uses_provider(self):
        conn = FakeNeo4jConnection()
        calls = []
        def tracking_embedder(text):
            calls.append(text)
            return [0.0] * 384

        brain = Neo4jBrain(connection=conn, embedder=tracking_embedder, preload_embeddings=False)
        brain._embed("hello world")
        assert calls == ["hello world"]

    def test_rejects_invalid_embedder(self):
        conn = FakeNeo4jConnection()
        with pytest.raises(TypeError):
            Neo4jBrain(connection=conn, embedder=42)

    def test_add_memory_uses_provider(self):
        conn = FakeNeo4jConnection()
        embed_called = False
        def tracking_embedder(text):
            nonlocal embed_called
            embed_called = True
            return [0.0] * 384

        brain = Neo4jBrain(connection=conn, embedder=tracking_embedder, preload_embeddings=False)
        brain.add_memory("Test content")
        assert embed_called, "add_memory should use the configured provider"

    def test_provider_error_propagates(self):
        """Embedding errors should propagate, not be silently swallowed."""
        conn = FakeNeo4jConnection()
        def failing_embedder(text):
            raise RuntimeError("GPU exploded")

        brain = Neo4jBrain(connection=conn, embedder=failing_embedder)
        with pytest.raises(EmbeddingError):
            brain._embed("test")


# ============================================================================
# PROTOCOL COMPLIANCE
# ============================================================================

class TestProtocolCompliance:
    """Verify that all providers satisfy the EmbeddingProvider protocol."""

    def test_callable_provider_is_embedding_provider(self):
        provider = CallableProvider(mock_embedder)
        assert isinstance(provider, EmbeddingProvider)

    def test_sentence_transformer_is_embedding_provider(self):
        provider = SentenceTransformerProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_openai_is_embedding_provider(self):
        provider = OpenAIProvider(api_key="sk-test")
        assert isinstance(provider, EmbeddingProvider)

    def test_ollama_is_embedding_provider(self):
        provider = OllamaProvider()
        assert isinstance(provider, EmbeddingProvider)

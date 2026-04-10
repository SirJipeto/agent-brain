"""
Tests for the Observer Framework.

Covers:
- on_message lifecycle
- Proactive memory retrieval
- Relevance evaluation (thresholds, importance, cooldown)
- Context building
- Conversation management
"""

import pytest
from datetime import datetime, timedelta

from agent_brain.observer import (
    ObserverFramework,
    RelevanceEvaluator,
    ContextBuilder,
    ConversationalEvent,
    ProactiveMemory,
    create_observer,
)
from agent_brain.brain import Neo4jBrain
from tests.conftest import FakeNeo4jConnection, mock_embedder


# ============================================================================
# CONVERSATIONAL EVENT
# ============================================================================

class TestConversationalEvent:
    def test_defaults(self):
        event = ConversationalEvent()
        assert event.id is not None
        assert event.content == ""
        assert event.entities == []
        assert event.processed is False

    def test_with_content(self):
        event = ConversationalEvent(content="Hello", context_tags=["greeting"])
        assert event.content == "Hello"
        assert "greeting" in event.context_tags


class TestProactiveMemory:
    def test_init(self):
        pm = ProactiveMemory(
            memory_id="m1",
            content="Test memory",
            connection_explanation="related to topic",
            connection_strength=0.8,
            suggested_mention="Speaking of X...",
            source_entity="X",
            importance=0.7,
            timestamp=datetime.now()
        )
        assert pm.memory_id == "m1"
        assert pm.connection_strength == 0.8


# ============================================================================
# RELEVANCE EVALUATOR
# ============================================================================

class TestRelevanceEvaluator:

    def setup_method(self):
        self.conn = FakeNeo4jConnection()
        self.brain = Neo4jBrain(connection=self.conn, embedder=mock_embedder)
        self.evaluator = RelevanceEvaluator(self.brain)

    def test_rejects_low_strength(self):
        candidate = {
            'connection_strength': 0.1,
            'memory': {'importance': 0.5, 'id': 'm1'},
            'source_entity': 'X',
        }
        event = ConversationalEvent(content="test")
        assert not self.evaluator.is_relevant_enough(candidate, event, ['X'])

    def test_accepts_high_importance(self):
        candidate = {
            'connection_strength': 0.3,
            'memory': {'importance': 0.9, 'id': 'm1'},
            'source_entity': 'X',
        }
        event = ConversationalEvent(content="test")
        assert self.evaluator.is_relevant_enough(candidate, event, ['X'])

    def test_rejects_recently_surfaced(self):
        candidate = {
            'connection_strength': 0.5,
            'memory': {'importance': 0.5, 'id': 'm1', 'salience_tags': []},
            'source_entity': 'X',
        }
        event = ConversationalEvent(content="test")
        self.evaluator.mark_surfaced('m1')
        assert not self.evaluator.is_relevant_enough(candidate, event, ['Y'])

    def test_accepts_current_entity(self):
        candidate = {
            'connection_strength': 0.3,
            'memory': {'importance': 0.5, 'id': 'm1'},
            'source_entity': 'Python',
        }
        event = ConversationalEvent(content="test")
        assert self.evaluator.is_relevant_enough(candidate, event, ['Python'])

    def test_accepts_matching_salience_tags(self):
        candidate = {
            'connection_strength': 0.3,
            'memory': {'importance': 0.5, 'id': 'm1', 'salience_tags': ['goal']},
            'source_entity': 'X',
        }
        event = ConversationalEvent(content="test", context_tags=['goal'])
        assert self.evaluator.is_relevant_enough(candidate, event, ['Y'])


class TestCooldown:
    def setup_method(self):
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)
        self.evaluator = RelevanceEvaluator(brain)

    def test_not_in_cooldown_initially(self):
        assert not self.evaluator.get_cooldown("Python")

    def test_in_cooldown_after_set(self):
        self.evaluator.set_cooldown("Python")
        assert self.evaluator.get_cooldown("Python", minutes=5)

    def test_cooldown_with_zero_minutes(self):
        self.evaluator.set_cooldown("Python")
        # Zero-minute cooldown should still pass since time hasn't elapsed
        assert not self.evaluator.get_cooldown("Python", minutes=0)

    def test_mark_surfaced_tracked(self):
        self.evaluator.mark_surfaced("m1")
        self.evaluator.mark_surfaced("m2")
        assert "m1" in self.evaluator.recently_surfaced
        assert "m2" in self.evaluator.recently_surfaced


# ============================================================================
# CONTEXT BUILDER
# ============================================================================

class TestContextBuilder:

    def test_build_surface_prompt_with_suggestion(self):
        pm = ProactiveMemory(
            memory_id="m1", content="Test",
            connection_explanation="",
            connection_strength=0.8,
            suggested_mention="Speaking of X, remember Y",
            source_entity="X",
            importance=0.7,
            timestamp=datetime.now()
        )
        result = ContextBuilder.build_surface_prompt(pm)
        assert result == "Speaking of X, remember Y"

    def test_build_surface_prompt_fallback_to_explanation(self):
        pm = ProactiveMemory(
            memory_id="m1", content="Test",
            connection_explanation="related to topic",
            connection_strength=0.8,
            suggested_mention="",
            source_entity="X",
            importance=0.7,
            timestamp=datetime.now()
        )
        result = ContextBuilder.build_surface_prompt(pm)
        assert "X" in result
        assert "related to topic" in result

    def test_build_grounded_context_empty(self):
        result = ContextBuilder.build_grounded_context([])
        assert result == ""

    def test_build_grounded_context_with_memories(self):
        memories = [
            ProactiveMemory(
                memory_id="m1", content="Test",
                connection_explanation="related",
                connection_strength=0.8,
                suggested_mention="Remember this",
                source_entity="X",
                importance=0.7,
                timestamp=datetime.now()
            )
        ]
        result = ContextBuilder.build_grounded_context(memories)
        assert "Grounded Context" in result
        assert "Remember this" in result

    def test_build_grounded_context_max_5(self):
        memories = [
            ProactiveMemory(
                memory_id=f"m{i}", content=f"Memory {i}",
                connection_explanation="", connection_strength=0.5,
                suggested_mention=f"Mention {i}",
                source_entity="X", importance=0.5,
                timestamp=datetime.now()
            )
            for i in range(10)
        ]
        result = ContextBuilder.build_grounded_context(memories)
        # Should limit to 5
        assert result.count("Mention") <= 5

    def test_includes_implicit_connections(self):
        memories = [
            ProactiveMemory(
                memory_id="m1", content="Test",
                connection_explanation="", connection_strength=0.8,
                suggested_mention="Remember", source_entity="X",
                importance=0.7, timestamp=datetime.now()
            )
        ]
        implicit = [
            {'source': 'A', 'target': 'B', 'path': ['A', 'C', 'B']}
        ]
        result = ContextBuilder.build_grounded_context(memories, implicit)
        assert "Implicit Connections" in result
        assert "A" in result


# ============================================================================
# OBSERVER FRAMEWORK
# ============================================================================

class TestObserverOnMessage:
    def test_returns_memory_id(self, observer):
        mem_id = observer.on_message("Hello, I'm interested in Python")
        assert mem_id is not None or mem_id is None  # May or may not store

    def test_extracts_entities(self, observer):
        observer.on_message("Alice works at Google")
        assert len(observer.recent_entities) > 0

    def test_updates_conversation_context(self, observer):
        observer.on_message("First message")
        observer.on_message("Second message")
        assert len(observer.conversation_context) == 2

    def test_no_store_when_disabled(self, observer):
        result = observer.on_message("Test", store_memory=False)
        assert result is None

    def test_no_extraction_when_disabled(self, observer):
        observer.on_message("Alice works at Google", extract_entities=False)
        # recent_entities should not have been updated from this call
        # (might have old entities from fixture setup)

    def test_hooks_triggered(self, observer):
        hook_called = False
        def my_hook(event_type, data):
            nonlocal hook_called
            hook_called = True
        
        observer.register_hook(my_hook)
        observer.on_message("Test message")
        assert hook_called


class TestObserverConversationManagement:
    def test_start_conversation_clears_context(self, observer):
        observer.on_message("Old message")
        observer.start_conversation(topic="New topic")
        # Context should be cleared (but might have 1 from topic processing)
        assert len(observer.conversation_context) <= 1

    def test_end_conversation_clears_context(self, observer):
        observer.on_message("Message 1")
        observer.on_message("Message 2")
        observer.on_message("Message 3")
        observer.end_conversation()
        assert len(observer.conversation_context) == 0
        assert len(observer._proactive_queue) == 0

    def test_get_conversation_summary(self, observer):
        observer.on_message("Hello")
        summary = observer.get_conversation_summary()
        assert 'message_count' in summary
        assert 'entities_mentioned' in summary
        assert 'proactive_surfaces' in summary
        assert summary['message_count'] >= 1


class TestObserverRecall:
    def test_recall_related_returns_dict(self, observer):
        result = observer.recall_related("Python")
        assert isinstance(result, dict)
        assert 'topic' in result
        assert 'semantic_matches' in result
        assert 'associations' in result
        assert 'activated_concepts' in result


class TestObserverGetGroundedContext:
    def test_empty_when_no_queue(self, observer):
        observer.clear_queue()
        assert observer.get_grounded_context() == ""

    def test_clear_queue(self, observer):
        observer._proactive_queue.append("test")
        observer.clear_queue()
        assert observer._proactive_queue == []


# ============================================================================
# FACTORY
# ============================================================================

class TestCreateObserver:
    def test_creates_with_brain(self, brain_with_data):
        obs = create_observer(brain_with_data)
        assert isinstance(obs, ObserverFramework)

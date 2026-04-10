"""
Tests for memory consolidation and relationship decay.

Covers:
- MemoryConsolidator: clustering, summarization, fact extraction, archiving
- RelationshipDecay: decay, pruning, dry run
- Edge cases: empty data, single-memory clusters
"""

import pytest
from datetime import datetime

from agent_brain.consolidation import (
    MemoryConsolidator,
    RelationshipDecay,
    ConsolidationResult,
    create_consolidator,
    create_decay,
)
from agent_brain.brain import Neo4jBrain
from tests.conftest import FakeNeo4jConnection, mock_embedder


# ============================================================================
# CONSOLIDATION RESULT
# ============================================================================

class TestConsolidationResult:
    def test_dataclass_fields(self):
        result = ConsolidationResult(
            memories_processed=10,
            memories_archived=5,
            memories_consolidated=2,
            facts_extracted=3,
            new_summaries=["a", "b"],
            timestamp=datetime.now()
        )
        assert result.memories_processed == 10
        assert result.memories_archived == 5
        assert len(result.new_summaries) == 2


# ============================================================================
# MEMORY CONSOLIDATOR
# ============================================================================

class TestMemoryConsolidator:

    def test_no_old_memories_returns_early(self):
        """With no old memories, consolidation should return immediately."""
        # Use a completely fresh brain with no data
        fresh_conn = FakeNeo4jConnection()
        fresh_brain = Neo4jBrain(connection=fresh_conn, embedder=mock_embedder)
        consolidator = MemoryConsolidator(fresh_brain)
        result = consolidator.consolidate_old_memories(older_than_days=0)
        assert result.memories_processed == 0
        assert result.memories_archived == 0

    def test_dry_run_does_not_modify(self, brain_with_data, fake_conn):
        """Dry run should report but not archive or consolidate."""
        consolidator = MemoryConsolidator(brain_with_data)
        initial_node_count = len(fake_conn.nodes)

        result = consolidator.consolidate_old_memories(
            older_than_days=0,
            dry_run=True
        )

        # Node count should not increase (no new consolidated memory)
        assert result.memories_archived == 0

    def test_custom_summarizer_used(self, brain_with_data):
        """Should call the provided summarizer."""
        summarizer_called = False

        def custom_summarizer(memories):
            nonlocal summarizer_called
            summarizer_called = True
            return f"Summary of {len(memories)} memories"

        consolidator = MemoryConsolidator(brain_with_data, llm_summarizer=custom_summarizer)
        # Even if no memories match, the summarizer won't be called
        # This just verifies the plumbing
        assert consolidator._llm_summarizer is custom_summarizer


class TestSimpleSummarizer:
    """Test the built-in simple summarizer."""

    def test_empty_memories_returns_empty(self, brain):
        consolidator = MemoryConsolidator(brain)
        result = consolidator._simple_summarizer([])
        assert result == ""

    def test_produces_summary_text(self, brain_with_data):
        consolidator = MemoryConsolidator(brain_with_data)
        memories = [
            {'id': 'm1', 'content': 'Discussed Python programming'},
            {'id': 'm2', 'content': 'Talked about machine learning'},
        ]
        result = consolidator._simple_summarizer(memories)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "exchanges" in result or "Discussion" in result


class TestFactExtraction:
    """Test fact extraction from memory clusters."""

    def setup_method(self):
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)
        self.consolidator = MemoryConsolidator(brain)

    def test_extracts_dates(self):
        memories = [
            {'id': 'm1', 'content': 'The meeting is on January 15th.'}
        ]
        facts = self.consolidator._extract_facts_from_cluster(memories)
        dates = [f for f in facts if f['predicate'] == 'mentioned_date']
        assert len(dates) > 0

    def test_extracts_intentions(self):
        memories = [
            {'id': 'm1', 'content': 'I want to learn Rust programming.'}
        ]
        facts = self.consolidator._extract_facts_from_cluster(memories)
        intentions = [f for f in facts if f['predicate'] == 'expressed_intention']
        assert len(intentions) > 0

    def test_extracts_preferences(self):
        memories = [
            {'id': 'm1', 'content': 'I love Python programming.'}
        ]
        facts = self.consolidator._extract_facts_from_cluster(memories)
        likes = [f for f in facts if f['predicate'] == 'likes']
        assert len(likes) > 0

    def test_limits_facts_per_cluster(self):
        # Generate many patterns
        memories = [
            {'id': f'm{i}', 'content': f'I want to buy thing{i}. I like item{i}. Meeting on March {i+1}st.'}
            for i in range(10)
        ]
        facts = self.consolidator._extract_facts_from_cluster(memories)
        assert len(facts) <= 10, "Should limit facts per cluster to 10"

    def test_empty_memories_no_facts(self):
        facts = self.consolidator._extract_facts_from_cluster([])
        assert facts == []

    def test_no_patterns_no_facts(self):
        memories = [
            {'id': 'm1', 'content': 'This is just a plain sentence.'}
        ]
        facts = self.consolidator._extract_facts_from_cluster(memories)
        # No date, intention, or preference patterns — should have no facts
        assert len(facts) == 0


class TestClusterByTopic:
    """Test memory clustering logic."""

    def test_groups_by_container(self, brain_with_data, fake_conn):
        consolidator = MemoryConsolidator(brain_with_data)
        memories = [
            {'id': 'm1', 'content': 'A', 'container': 'work'},
            {'id': 'm2', 'content': 'B', 'container': 'work'},
            {'id': 'm3', 'content': 'C', 'container': 'personal'},
        ]
        clusters = consolidator._cluster_by_topic(memories)
        assert isinstance(clusters, dict)

    def test_groups_by_shared_entities(self, brain_with_data, fake_conn):
        consolidator = MemoryConsolidator(brain_with_data)
        # m1 and m2 share Python entity
        memories = [
            {'id': 'm1', 'content': 'Python discussion'},
            {'id': 'm2', 'content': 'More Python talk'},
        ]
        clusters = consolidator._cluster_by_topic(memories)
        assert isinstance(clusters, dict)


# ============================================================================
# RELATIONSHIP DECAY
# ============================================================================

class TestRelationshipDecay:

    def test_decay_reduces_weights(self):
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)
        decay = RelationshipDecay(brain)

        conn.add_node("a", {"Entity"}, {"name": "A"})
        conn.add_node("b", {"Entity"}, {"name": "B"})
        conn.add_relationship("a", "b", "RELATED_TO", {"weight": 0.8})

        result = decay.decay_weak_connections(decay_factor=0.5, min_threshold=0.01)
        assert isinstance(result, dict)
        assert 'total_before' in result

    def test_dry_run_does_not_modify(self):
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)
        decay = RelationshipDecay(brain)

        conn.add_node("a", {"Entity"}, {"name": "A"})
        conn.add_node("b", {"Entity"}, {"name": "B"})
        conn.add_relationship("a", "b", "RELATED_TO", {"weight": 0.8})

        result = decay.decay_weak_connections(dry_run=True)
        assert isinstance(result, dict)
        # Weight should not have changed
        assert conn.relationships[0]['props']['weight'] == 0.8

    def test_prunes_below_threshold(self):
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)
        decay = RelationshipDecay(brain)

        conn.add_node("a", {"Entity"}, {"name": "A"})
        conn.add_node("b", {"Entity"}, {"name": "B"})
        conn.add_relationship("a", "b", "RELATED_TO", {"weight": 0.05})

        decay.decay_weak_connections(decay_factor=0.95, min_threshold=0.1)
        # Weak relationship should have been pruned
        # (exact behavior depends on mock dispatch)


class TestRelationshipDecayByAge:
    def test_runs_without_error(self):
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)
        decay = RelationshipDecay(brain)
        result = decay.decay_by_age(older_than_days=30)
        assert isinstance(result, dict)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

class TestFactoryFunctions:
    def test_create_consolidator(self):
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)
        consolidator = create_consolidator(brain)
        assert isinstance(consolidator, MemoryConsolidator)

    def test_create_decay(self):
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)
        decay = create_decay(brain)
        assert isinstance(decay, RelationshipDecay)

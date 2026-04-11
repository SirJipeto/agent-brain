"""
Tests for Neo4jBrain core operations.

Covers:
- add_memory (with and without entity extraction)
- semantic_search
- graph_traverse
- hybrid_search
- relate_entities (including ALLOWED_RELATIONSHIPS whitelist)
- Maintenance operations (decay, archive, stats)
"""

import pytest
from datetime import datetime

from agent_brain.brain import Neo4jBrain
from tests.conftest import FakeNeo4jConnection, mock_embedder



# ============================================================================
# INITIALIZATION
# ============================================================================

class TestNeo4jBrainInit:
    def test_creates_with_provided_connection(self, fake_conn):
        brain = Neo4jBrain(connection=fake_conn, embedder=mock_embedder)
        assert brain.conn is fake_conn

    def test_uses_provided_embedder(self, fake_conn):
        custom_called = False
        def custom_embedder(text):
            nonlocal custom_called
            custom_called = True
            return [0.0] * 384

        brain = Neo4jBrain(connection=fake_conn, embedder=custom_embedder)
        brain._embed("test")
        assert custom_called

    def test_schema_creation_runs(self, fake_conn):
        """_ensure_schema should run constraint queries during init"""
        brain = Neo4jBrain(connection=fake_conn, embedder=mock_embedder)
        # Check that constraint queries were logged
        writes = [log for log in fake_conn.get_call_log() if log[0] == 'write']
        assert len(writes) >= 3, "Should have run constraint creation queries"


# ============================================================================
# MEMORY OPERATIONS
# ============================================================================

class TestAddMemory:
    def test_returns_memory_id(self, brain):
        mem_id = brain.add_memory("Test content")
        assert isinstance(mem_id, str)
        assert len(mem_id) > 0

    def test_stores_memory_in_connection(self, brain, fake_conn):
        brain.add_memory("Hello world")
        memories = fake_conn.get_nodes_by_label("Memory")
        assert len(memories) >= 1

    def test_auto_extracts_entities(self, brain, fake_conn):
        brain.add_memory("Alice went to Paris")
        # Should have extracted entities via extract_entities
        call_log = fake_conn.get_call_log()
        # Entity merges should appear in the log
        entity_writes = [
            log for log in call_log
            if log[0] == 'write' and 'MERGE' in log[1] and 'Entity' in log[1]
        ]
        assert len(entity_writes) > 0, "Should have merged entity nodes"

    def test_skips_extraction_when_disabled(self, brain, fake_conn):
        brain.add_memory("Alice went to Paris", extract_entities=False)
        call_log = fake_conn.get_call_log()
        entity_writes = [
            log for log in call_log
            if log[0] == 'write' and 'MERGE' in log[1] and 'Entity' in log[1]
        ]
        assert len(entity_writes) == 0, "Should not extract entities when disabled"

    def test_uses_pre_extracted_entities(self, brain, fake_conn):
        entities = [
            {'name': 'CustomEntity', 'type': 'technology', 'description': 'test'}
        ]
        brain.add_memory("Some content", entities=entities)
        # Should have merged CustomEntity
        call_log = fake_conn.get_call_log()
        entity_params = [
            log[2] for log in call_log
            if log[0] == 'write' and 'MERGE' in log[1] and log[2].get('name') == 'CustomEntity'
        ]
        assert len(entity_params) > 0

    def test_summary_defaults_to_content_truncated(self, brain, fake_conn):
        long_content = "x" * 500
        brain.add_memory(long_content)
        memories = fake_conn.get_nodes_by_label("Memory")
        assert len(memories) >= 1
        # Summary param should be content[:200]
        call_log = fake_conn.get_call_log()
        create_writes = [
            log for log in call_log
            if log[0] == 'write' and 'CREATE' in log[1] and 'Memory' in log[1]
        ]
        assert len(create_writes) >= 1
        assert len(create_writes[0][2].get('summary', '')) <= 200

    def test_custom_params(self, brain, fake_conn):
        brain.add_memory(
            "Test",
            summary="My summary",
            content_type="note",
            importance=0.9,
            salience_tags=["goal"],
            container="work",
            source="manual",
            metadata={"key": "value"},
        )
        call_log = fake_conn.get_call_log()
        # Filter for CREATE Memory queries, excluding schema constraints and indexes
        create_writes = [
            log for log in call_log
            if log[0] == 'write'
            and 'CREATE' in log[1]
            and 'Memory' in log[1]
            and 'CONSTRAINT' not in log[1]
            and 'INDEX' not in log[1]
        ]
        assert len(create_writes) >= 1
        params = create_writes[0][2]
        assert params.get('summary') == "My summary"
        assert params.get('content_type') == "note"
        assert params.get('importance') == 0.9
        assert params.get('container') == "work"


# ============================================================================
# SEARCH OPERATIONS
# ============================================================================

class TestSemanticSearch:
    def test_returns_list(self, brain_with_data):
        results = brain_with_data.semantic_search("machine learning")
        assert isinstance(results, list)

    def test_respects_top_k(self, brain_with_data):
        results = brain_with_data.semantic_search("test", top_k=1)
        assert len(results) <= 1

    def test_filters_by_container(self, brain_with_data):
        results = brain_with_data.semantic_search("test", container="nonexistent")
        assert isinstance(results, list)

    def test_fallback_to_text_search(self, brain):
        """When vector search fails, should fall back to text search"""
        # With no data, semantic search should not crash
        results = brain.semantic_search("anything")
        assert isinstance(results, list)


class TestGraphTraverse:
    def test_returns_connections(self, brain_with_data):
        results = brain_with_data.graph_traverse("Python")
        assert isinstance(results, list)
        if results:
            assert 'connected_name' in results[0]
            assert 'strength' in results[0]

    def test_nonexistent_entity_returns_empty(self, brain_with_data):
        results = brain_with_data.graph_traverse("NonexistentEntity")
        assert results == []

    def test_filters_by_relationship_type(self, brain_with_data):
        results = brain_with_data.graph_traverse("Python", relationship_type="RELATED_TO")
        assert isinstance(results, list)


class TestHybridSearch:
    def test_returns_dict_with_expected_keys(self, brain_with_data):
        result = brain_with_data.hybrid_search("Python programming")
        assert isinstance(result, dict)
        assert 'query' in result
        assert 'semantic_matches' in result
        assert 'graph_insights' in result
        assert 'related_entities' in result
        assert 'timestamp' in result

    def test_query_preserved(self, brain_with_data):
        result = brain_with_data.hybrid_search("my query")
        assert result['query'] == "my query"


# ============================================================================
# ASSOCIATIVE OPERATIONS
# ============================================================================

class TestRecallAssociations:
    def test_returns_list(self, brain_with_data):
        results = brain_with_data.recall_associations("Python")
        assert isinstance(results, list)


class TestSpreadActivation:
    def test_returns_dict(self, brain_with_data):
        result = brain_with_data.spread_activation(["Python"])
        assert isinstance(result, dict)

    def test_seed_not_in_results(self, brain_with_data):
        result = brain_with_data.spread_activation(["Python"])
        assert "Python" not in result, "Seed entities should not appear in results"

    def test_multiple_seeds(self, brain_with_data):
        result = brain_with_data.spread_activation(["Python", "Neo4j"])
        assert "Python" not in result
        assert "Neo4j" not in result

    def test_threshold_filtering(self, brain_with_data):
        result = brain_with_data.spread_activation(["Python"], threshold=0.9)
        # With high threshold, fewer results expected
        for name, activation in result.items():
            assert activation >= 0.9

    def test_empty_seeds_returns_empty(self, brain_with_data):
        result = brain_with_data.spread_activation([])
        assert result == {}


class TestFindImplicitConnections:
    def test_returns_list(self, brain_with_data):
        result = brain_with_data.find_implicit_connections("Python", "Neo4j")
        assert isinstance(result, list)


class TestGetRelatedMemories:
    def test_returns_memories(self, brain_with_data):
        results = brain_with_data.get_related_memories("Python")
        assert isinstance(results, list)

    def test_nonexistent_entity_returns_empty(self, brain_with_data):
        results = brain_with_data.get_related_memories("NonexistentEntity")
        assert results == []


# ============================================================================
# RELATIONSHIP OPERATIONS
# ============================================================================

class TestRelateEntities:
    def test_creates_relationship(self, brain, fake_conn):
        brain.relate_entities("A", "B", "RELATED_TO", weight=0.8)
        call_log = fake_conn.get_call_log()
        merge_writes = [log for log in call_log if log[0] == 'write' and 'RELATED_TO' in log[1]]
        assert len(merge_writes) > 0

    def test_blocks_disallowed_relationship(self, brain, fake_conn):
        initial_count = len(fake_conn.get_call_log())
        brain.relate_entities("A", "B", "HACKS_INTO")
        # Should not have made any write calls
        assert len(fake_conn.get_call_log()) == initial_count

    def test_auto_discovers_relationship(self, brain, fake_conn):
        brain.relate_entities("A", "B", "INVENTED", auto_discover=True)
        assert "INVENTED" in brain.allowed_relationships
        
        call_log = fake_conn.get_call_log()
        config_writes = [log for log in call_log if log[0] == 'write' and 'custom_relationship_types' in log[1]]
        assert len(config_writes) > 0
        
        merge_writes = [log for log in call_log if log[0] == 'write' and 'INVENTED' in log[1]]
        assert len(merge_writes) > 0

    def test_register_relationship_type(self, brain, fake_conn):
        brain.register_relationship_type("LOVES pizza")
        assert "LOVES_PIZZA" in brain.allowed_relationships
        
        call_log = fake_conn.get_call_log()
        config_writes = [log for log in call_log if log[0] == 'write' and 'custom_relationship_types' in log[1]]
        assert len(config_writes) > 0

    def test_case_insensitive_relationship_check(self, brain, fake_conn):
        brain.relate_entities("A", "B", "related_to")
        call_log = fake_conn.get_call_log()
        merge_writes = [log for log in call_log if log[0] == 'write' and 'RELATED_TO' in log[1]]
        assert len(merge_writes) > 0, "Should uppercase and accept valid rel types"

    def test_all_allowed_relationships_accepted(self, brain, fake_conn):
        for rel in Neo4jBrain.ALLOWED_RELATIONSHIPS:
            fake_conn._call_log.clear()  # Reset log
            brain.relate_entities("X", "Y", rel)
            writes = [log for log in fake_conn.get_call_log() if log[0] == 'write']
            assert len(writes) > 0, f"Relationship type {rel} should be allowed"


class TestStrengthenRelationship:
    def test_boosts_weight(self, brain_with_data, fake_conn):
        brain_with_data.strengthen_relationship("Python", "Machine Learning", boost=0.1)
        call_log = fake_conn.get_call_log()
        writes = [log for log in call_log if log[0] == 'write']
        assert len(writes) > 0


# ============================================================================
# MAINTENANCE
# ============================================================================




class TestArchiveOldMemories:
    def test_runs_without_error(self, brain_with_data):
        result = brain_with_data.archive_old_memories(older_than_days=0)
        assert isinstance(result, int)


class TestGetStats:
    def test_returns_expected_keys(self, brain_with_data):
        stats = brain_with_data.get_stats()
        assert 'memories' in stats
        assert 'entities' in stats
        assert 'relationships' in stats

    def test_counts_are_non_negative(self, brain_with_data):
        stats = brain_with_data.get_stats()
        assert stats['memories'] >= 0
        assert stats['entities'] >= 0
        assert stats['relationships'] >= 0


class TestGetUserGoals:
    def test_returns_goals(self, brain_with_data):
        goals = brain_with_data.get_user_goals()
        assert isinstance(goals, list)

    def test_empty_when_no_person(self, brain):
        goals = brain.get_user_goals()
        assert goals == []

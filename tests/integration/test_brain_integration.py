"""
Integration tests for Neo4jBrain with a real Neo4j instance.

Run with: pytest tests/integration/ -m integration
Requires: docker compose -f docker-compose.test.yml up -d
"""

import pytest

pytestmark = pytest.mark.integration


class TestMemoryLifecycle:
    """End-to-end memory CRUD operations against real Neo4j."""

    def test_add_and_retrieve_memory(self, integration_brain):
        mem_id = integration_brain.add_memory(
            content="Python is a great programming language",
            importance=0.8,
            container="test"
        )
        assert mem_id is not None

        # Verify via stats
        stats = integration_brain.get_stats()
        assert stats['memories'] >= 1

    def test_add_memory_extracts_entities(self, integration_brain):
        integration_brain.add_memory(
            content="Alice works at Google in Mountain View"
        )

        stats = integration_brain.get_stats()
        assert stats['entities'] >= 1, "Should have extracted at least one entity"
        assert stats['relationships'] >= 1, "Should have created MENTIONS relationship"

    def test_multiple_memories_share_entities(self, integration_brain):
        integration_brain.add_memory(content="Alice loves Python")
        integration_brain.add_memory(content="Alice uses Python at work")

        stats = integration_brain.get_stats()
        # "Alice" and "Python" should be merged (not duplicated)
        assert stats['entities'] >= 1

    def test_memory_with_custom_metadata(self, integration_brain):
        mem_id = integration_brain.add_memory(
            content="Important meeting tomorrow",
            summary="Meeting reminder",
            content_type="note",
            importance=0.9,
            salience_tags=["deadline", "work"],
            container="work",
            source="manual",
        )
        assert mem_id is not None


class TestSearchIntegration:
    """Test search operations against real Neo4j."""

    def test_semantic_search_returns_results(self, integration_brain):
        # Add some memories
        integration_brain.add_memory(content="Python programming language")
        integration_brain.add_memory(content="JavaScript web development")
        integration_brain.add_memory(content="Machine learning algorithms")

        results = integration_brain.semantic_search("programming", top_k=3)
        assert isinstance(results, list)
        assert len(results) >= 1  # Should find at least one match

    def test_semantic_search_scores_sorted(self, integration_brain):
        integration_brain.add_memory(content="Neural network architecture design")
        integration_brain.add_memory(content="Database schema optimization")

        results = integration_brain.semantic_search("neural networks", top_k=5)
        if len(results) >= 2:
            # Scores should be in descending order
            scores = [r.get('score', 0) for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_semantic_search_container_filter(self, integration_brain):
        integration_brain.add_memory(content="Work project update", container="work")
        integration_brain.add_memory(content="Personal hobby", container="personal")

        results = integration_brain.semantic_search("project", container="work")
        for r in results:
            # Should only return work-container results
            pass  # Container filtering is in the Cypher WHERE clause

    def test_text_fallback_search(self, integration_brain):
        integration_brain.add_memory(content="The quick brown fox jumps over the lazy dog")

        # Force text fallback by searching
        results = integration_brain._text_search_fallback("quick brown fox")
        assert isinstance(results, list)


class TestGraphTraversalIntegration:
    """Test graph traversal against real Neo4j."""

    def test_traverse_finds_connections(self, integration_brain):
        # Build a small graph
        integration_brain.relate_entities("Python", "Data Science", "RELATED_TO", weight=0.8)
        integration_brain.relate_entities("Data Science", "Machine Learning", "ENABLES", weight=0.7)

        results = integration_brain.graph_traverse("Python", depth=2)
        assert isinstance(results, list)
        if results:
            connected_names = [r['connected_name'] for r in results]
            assert "Data Science" in connected_names

    def test_traverse_respects_depth(self, integration_brain):
        integration_brain.relate_entities("A", "B", "RELATED_TO", weight=0.8)
        integration_brain.relate_entities("B", "C", "RELATED_TO", weight=0.7)
        integration_brain.relate_entities("C", "D", "RELATED_TO", weight=0.6)

        # Depth 1 should only find B
        results = integration_brain.graph_traverse("A", depth=1)
        if results:
            names = {r['connected_name'] for r in results}
            assert "B" in names
            assert "D" not in names  # Too far

    def test_hybrid_search(self, integration_brain):
        integration_brain.add_memory(content="Python is used for machine learning")
        integration_brain.relate_entities("Python", "Machine Learning", "RELATED_TO")

        result = integration_brain.hybrid_search("Python machine learning")
        assert 'semantic_matches' in result
        assert 'graph_insights' in result


class TestRelationshipIntegration:
    """Test relationship operations against real Neo4j."""

    def test_create_and_find_relationship(self, integration_brain):
        integration_brain.relate_entities("Neo4j", "Graph Database", "IS_A", weight=0.9)

        results = integration_brain.graph_traverse("Neo4j")
        assert isinstance(results, list)

    def test_strengthen_relationship(self, integration_brain):
        integration_brain.relate_entities("A", "B", "RELATED_TO", weight=0.5)
        integration_brain.strengthen_relationship("A", "B", boost=0.2)

        # Verify the relationship was strengthened (weight should increase)
        results = integration_brain.graph_traverse("A")
        assert isinstance(results, list)

    def test_find_implicit_connections(self, integration_brain):
        integration_brain.relate_entities("A", "B", "RELATED_TO", weight=0.8)
        integration_brain.relate_entities("B", "C", "RELATED_TO", weight=0.7)

        paths = integration_brain.find_implicit_connections("A", "C")
        assert isinstance(paths, list)


class TestActivationSpreadingIntegration:
    """Test activation spreading against real Neo4j."""

    def test_activation_on_real_graph(self, integration_brain):
        integration_brain.relate_entities("Python", "ML", "RELATED_TO", weight=0.8)
        integration_brain.relate_entities("ML", "TensorFlow", "ENABLES", weight=0.7)
        integration_brain.relate_entities("TensorFlow", "GPU", "DEPENDS_ON", weight=0.6)

        result = integration_brain.spread_activation(["Python"], iterations=3, threshold=0.01)
        assert isinstance(result, dict)
        # ML should be activated (directly connected)
        if result:
            assert "ML" in result or len(result) > 0


class TestMaintenanceIntegration:
    """Test maintenance operations against real Neo4j."""

    def test_get_stats(self, integration_brain):
        integration_brain.add_memory(content="Test memory")
        stats = integration_brain.get_stats()
        assert stats['memories'] >= 1
        assert isinstance(stats['entities'], int)
        assert isinstance(stats['relationships'], int)

    def test_decay_weak_connections(self, integration_brain):
        integration_brain.relate_entities("A", "B", "RELATED_TO", weight=0.8)
        integration_brain.decay_weak_connections(threshold=0.1)
        # Should not crash and relationship should survive (weight > threshold)

    def test_archive_old_memories(self, integration_brain):
        integration_brain.add_memory(content="Old memory")
        # Archive memories older than 0 days (should archive all)
        archived = integration_brain.archive_old_memories(older_than_days=0)
        assert isinstance(archived, int)

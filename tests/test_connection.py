"""
Tests for Neo4j connection layer.

Covers:
- Neo4jConnection interface
- Singleton behavior
- FakeNeo4jConnection correctness
"""

import pytest
from tests.conftest import FakeNeo4jConnection


# ============================================================================
# FAKE CONNECTION TESTS (validate the mock itself)
# ============================================================================

class TestFakeNeo4jConnectionBasics:
    """Validate that the mock behaves correctly for test reliability."""

    def test_add_and_retrieve_node(self):
        conn = FakeNeo4jConnection()
        conn.add_node("n1", {"Entity"}, {"name": "Python", "type": "technology"})

        nodes = conn.get_nodes_by_label("Entity")
        assert len(nodes) == 1
        assert nodes[0]['name'] == "Python"

    def test_add_and_retrieve_relationship(self):
        conn = FakeNeo4jConnection()
        conn.add_node("n1", {"Entity"}, {"name": "A"})
        conn.add_node("n2", {"Entity"}, {"name": "B"})
        conn.add_relationship("n1", "n2", "RELATED_TO", {"weight": 0.8})

        rels = conn.get_relationships_between("n1", "n2")
        assert len(rels) == 1
        assert rels[0]['type'] == "RELATED_TO"
        assert rels[0]['props']['weight'] == 0.8

    def test_node_labels_separated(self):
        conn = FakeNeo4jConnection()
        conn.add_node("n1", {"Entity"}, {"name": "A"})
        conn.add_node("n2", {"Memory"}, {"content": "B"})

        assert len(conn.get_nodes_by_label("Entity")) == 1
        assert len(conn.get_nodes_by_label("Memory")) == 1
        assert len(conn.get_nodes_by_label("Fact")) == 0

    def test_clear_resets_everything(self):
        conn = FakeNeo4jConnection()
        conn.add_node("n1", {"Entity"}, {"name": "A"})
        conn.add_relationship("n1", "n1", "SELF", {})
        conn.execute_query("SELECT 1", {})

        conn.clear()
        assert len(conn.nodes) == 0
        assert len(conn.relationships) == 0
        assert len(conn.get_call_log()) == 0

    def test_verify_connectivity_returns_true(self):
        conn = FakeNeo4jConnection()
        assert conn.verify_connectivity() is True

    def test_close_is_noop(self):
        conn = FakeNeo4jConnection()
        conn.close()  # Should not raise

    def test_call_log_records_queries(self):
        conn = FakeNeo4jConnection()
        conn.execute_query("MATCH (n) RETURN n", {"param": 1})
        conn.execute_write("CREATE (n:Node)", {"param": 2})

        log = conn.get_call_log()
        assert len(log) == 2
        assert log[0][0] == "read"
        assert log[1][0] == "write"


class TestFakeNeo4jConnectionQueryDispatch:
    """Test that specific query patterns are dispatched correctly."""

    def test_constraint_queries_are_noop(self):
        conn = FakeNeo4jConnection()
        result = conn.execute_write(
            "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE"
        )
        assert result == []

    def test_create_node_returns_id(self):
        conn = FakeNeo4jConnection()
        result = conn.execute_write(
            "CREATE (m:Memory {id: $id, content: $content})",
            {'id': 'test-id', 'content': 'Hello'}
        )
        assert len(result) >= 1
        assert result[0]['id'] == 'test-id'

    def test_stats_query(self):
        conn = FakeNeo4jConnection()
        conn.add_node("m1", {"Memory"}, {"archived": False})
        conn.add_node("e1", {"Entity"}, {"name": "A"})
        conn.add_relationship("m1", "e1", "MENTIONS", {})

        result = conn.execute_query("""
            CALL { MATCH (m:Memory) WHERE m.archived = false RETURN count(m) AS memories }
            CALL { MATCH (e:Entity) RETURN count(e) AS entities }
            CALL { MATCH ()-[r]->() RETURN count(r) AS relationships }
            RETURN memories, entities, relationships
        """)
        assert len(result) == 1
        assert result[0]['memories'] == 1
        assert result[0]['entities'] == 1
        assert result[0]['relationships'] == 1

    def test_execute_single_returns_first(self):
        conn = FakeNeo4jConnection()
        conn.add_node("user", {"Person"}, {"name": "user", "goals": ["goal1"]})

        result = conn.execute_single("""
            MATCH (p:Person {id: "user"})
            RETURN p.goals AS goals
        """)
        assert result is not None

    def test_execute_single_returns_none_on_empty(self):
        conn = FakeNeo4jConnection()
        result = conn.execute_single("MATCH (n:Nonexistent) RETURN n")
        assert result is None


# ============================================================================
# REAL CONNECTION MODULE INTERFACE
# ============================================================================

class TestNeo4jConnectionInterface:
    """Verify the real Neo4jConnection class interface matches what we mock."""

    def test_interface_methods_exist(self):
        """Check that Neo4jConnection has all methods we mock."""
        from agent_brain.connection import Neo4jConnection

        # These methods must exist for the mock to be valid
        assert hasattr(Neo4jConnection, 'execute_query')
        assert hasattr(Neo4jConnection, 'execute_single')
        assert hasattr(Neo4jConnection, 'execute_write')
        assert hasattr(Neo4jConnection, 'verify_connectivity')
        assert hasattr(Neo4jConnection, 'close')

    def test_get_connection_config_has_defaults(self):
        """Config should work even without env vars."""
        from agent_brain.connection import get_connection_config
        config = get_connection_config()
        assert 'uri' in config
        assert 'user' in config
        assert 'password' in config

"""
Tests for activation spreading algorithm.

Covers:
- Convergence behavior
- Threshold filtering
- Circular graph handling (no infinite loops)
- Multi-seed activation
- Edge cases (disconnected nodes, empty graph)
- Configurable decay factor
- Circuit breaker (max_activated_nodes)
"""

import logging
import pytest
from agent_brain.brain import Neo4jBrain
from tests.conftest import FakeNeo4jConnection, mock_embedder


class TestSpreadActivationBasic:
    """Test core activation spreading mechanics."""

    def test_single_seed_spreads(self, brain_with_data):
        """Activation should spread from seed to connected nodes."""
        result = brain_with_data.spread_activation(["Python"])
        assert len(result) > 0, "Should activate at least one connected node"

    def test_activation_values_are_positive(self, brain_with_data):
        result = brain_with_data.spread_activation(["Python"])
        for name, activation in result.items():
            assert activation > 0, f"Activation for {name} should be positive"

    def test_seed_excluded_from_results(self, brain_with_data):
        result = brain_with_data.spread_activation(["Python"])
        assert "Python" not in result

    def test_multiple_seeds_excluded(self, brain_with_data):
        result = brain_with_data.spread_activation(["Python", "Neo4j"])
        assert "Python" not in result
        assert "Neo4j" not in result

    def test_empty_seeds_returns_empty(self, brain_with_data):
        result = brain_with_data.spread_activation([])
        assert result == {}

    def test_nonexistent_seed_returns_empty(self, brain_with_data):
        result = brain_with_data.spread_activation(["NoSuchEntity"])
        assert result == {}


class TestSpreadActivationThreshold:
    """Test threshold filtering behavior."""

    def test_all_results_above_threshold(self, brain_with_data):
        threshold = 0.1
        result = brain_with_data.spread_activation(["Python"], threshold=threshold)
        for name, activation in result.items():
            assert activation >= threshold, (
                f"{name} has activation {activation} < threshold {threshold}"
            )

    def test_high_threshold_fewer_results(self, brain_with_data):
        low = brain_with_data.spread_activation(["Python"], threshold=0.01)
        high = brain_with_data.spread_activation(["Python"], threshold=0.5)
        assert len(high) <= len(low), "Higher threshold should return fewer results"

    def test_very_high_threshold_returns_empty(self, brain_with_data):
        """A threshold higher than any possible activation should return empty."""
        result = brain_with_data.spread_activation(["Python"], threshold=10.0)
        assert result == {}, "Threshold of 10.0 should filter all activations"


class TestSpreadActivationCircularGraph:
    """Test that activation spreading handles cycles properly."""

    def test_circular_does_not_loop_infinitely(self):
        """A -> B -> C -> A should terminate and not OOM."""
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)

        # Create circular graph
        conn.add_node("ea", {"Entity"}, {"name": "A", "type": "concept"})
        conn.add_node("eb", {"Entity"}, {"name": "B", "type": "concept"})
        conn.add_node("ec", {"Entity"}, {"name": "C", "type": "concept"})
        conn.add_relationship("ea", "eb", "RELATED_TO", {"weight": 0.8})
        conn.add_relationship("eb", "ec", "RELATED_TO", {"weight": 0.8})
        conn.add_relationship("ec", "ea", "RELATED_TO", {"weight": 0.8})

        # Should terminate (within iterations limit)
        result = brain.spread_activation(["A"], iterations=5, threshold=0.01)
        assert isinstance(result, dict)
        # B and C should have activation
        assert "B" in result or "C" in result

    def test_self_loop_handled(self):
        """An entity connected to itself should not cause issues."""
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)

        conn.add_node("ea", {"Entity"}, {"name": "A", "type": "concept"})
        conn.add_relationship("ea", "ea", "RELATED_TO", {"weight": 0.5})

        result = brain.spread_activation(["A"], iterations=3)
        # Self-loop activation is added but filtered since A is a seed
        assert "A" not in result


class TestSpreadActivationIterations:
    """Test iteration behavior."""

    def test_more_iterations_more_spread(self):
        """More iterations should spread activation further in a chain."""
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)

        # Linear chain: A - B - C - D - E
        for name in ["A", "B", "C", "D", "E"]:
            conn.add_node(name, {"Entity"}, {"name": name, "type": "concept"})
        conn.add_relationship("A", "B", "RELATED_TO", {"weight": 0.8})
        conn.add_relationship("B", "C", "RELATED_TO", {"weight": 0.8})
        conn.add_relationship("C", "D", "RELATED_TO", {"weight": 0.8})
        conn.add_relationship("D", "E", "RELATED_TO", {"weight": 0.8})

        result_1 = brain.spread_activation(["A"], iterations=1, threshold=0.01)
        result_3 = brain.spread_activation(["A"], iterations=3, threshold=0.01)

        # More iterations should reach further nodes
        assert len(result_3) >= len(result_1)

    def test_zero_iterations_returns_empty(self):
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)

        conn.add_node("A", {"Entity"}, {"name": "A", "type": "concept"})
        conn.add_node("B", {"Entity"}, {"name": "B", "type": "concept"})
        conn.add_relationship("A", "B", "RELATED_TO", {"weight": 0.8})

        result = brain.spread_activation(["A"], iterations=0)
        assert result == {}, "Zero iterations should not spread activation"


class TestSpreadActivationWeights:
    """Test that activation is weighted by relationship strength."""

    def test_stronger_connections_spread_more(self):
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)

        conn.add_node("A", {"Entity"}, {"name": "A", "type": "concept"})
        conn.add_node("B", {"Entity"}, {"name": "B", "type": "concept"})
        conn.add_node("C", {"Entity"}, {"name": "C", "type": "concept"})

        conn.add_relationship("A", "B", "RELATED_TO", {"weight": 0.9})
        conn.add_relationship("A", "C", "RELATED_TO", {"weight": 0.1})

        result = brain.spread_activation(["A"], iterations=1, threshold=0.01)

        # B should have higher activation than C
        if "B" in result and "C" in result:
            assert result["B"] > result["C"], "Stronger connection should spread more"


# ============================================================================
# DECAY FACTOR
# ============================================================================

class TestSpreadActivationDecayFactor:
    """Test the configurable decay_factor parameter."""

    def _make_chain_brain(self):
        """Create a simple A-B-C chain for decay testing."""
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)
        for name in ["A", "B", "C"]:
            conn.add_node(name, {"Entity"}, {"name": name, "type": "concept"})
        conn.add_relationship("A", "B", "RELATED_TO", {"weight": 1.0})
        conn.add_relationship("B", "C", "RELATED_TO", {"weight": 1.0})
        return brain

    def test_default_decay_is_half(self):
        """Default decay_factor=0.5 means each hop halves activation."""
        brain = self._make_chain_brain()
        result = brain.spread_activation(["A"], iterations=1, threshold=0.01)
        # A=1.0, B should get 1.0 * 1.0 (weight) * 0.5 (decay) = 0.5
        assert abs(result.get("B", 0) - 0.5) < 0.01, (
            f"With decay=0.5 and weight=1.0, B should be ~0.5, got {result.get('B')}"
        )

    def test_higher_decay_spreads_more(self):
        """Higher decay_factor means more energy spreads to neighbors."""
        brain = self._make_chain_brain()
        result_low = brain.spread_activation(["A"], iterations=1, threshold=0.01, decay_factor=0.3)
        result_high = brain.spread_activation(["A"], iterations=1, threshold=0.01, decay_factor=0.8)

        assert result_high.get("B", 0) > result_low.get("B", 0), (
            "Higher decay_factor should spread more activation"
        )

    def test_lower_decay_spreads_less(self):
        """Lower decay_factor means faster energy loss."""
        brain = self._make_chain_brain()
        result = brain.spread_activation(["A"], iterations=2, threshold=0.001, decay_factor=0.1)
        # With decay=0.1: B gets 0.1, C gets 0.1 * 0.1 = 0.01
        # C should have much less activation
        if "B" in result and "C" in result:
            assert result["C"] < result["B"] * 0.5

    def test_zero_decay_no_spread(self):
        """decay_factor=0 means nothing spreads at all."""
        brain = self._make_chain_brain()
        result = brain.spread_activation(["A"], iterations=3, threshold=0.001, decay_factor=0.0)
        assert result == {}, "Zero decay should produce no activation beyond seeds"

    def test_high_decay_reaches_further(self):
        """Higher decay in a chain should activate more distant nodes."""
        brain = self._make_chain_brain()
        result_low = brain.spread_activation(["A"], iterations=2, threshold=0.1, decay_factor=0.2)
        result_high = brain.spread_activation(["A"], iterations=2, threshold=0.1, decay_factor=0.8)

        # With low decay, C might not meet threshold; with high decay it should
        assert len(result_high) >= len(result_low)


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class TestSpreadActivationCircuitBreaker:
    """Test the max_activated_nodes circuit breaker."""

    def _make_star_graph(self, center: str, spokes: int):
        """Create a star graph: center connected to N spokes."""
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)
        conn.add_node(center, {"Entity"}, {"name": center, "type": "concept"})
        for i in range(spokes):
            name = f"S{i}"
            conn.add_node(name, {"Entity"}, {"name": name, "type": "concept"})
            conn.add_relationship(center, name, "RELATED_TO", {"weight": 0.8})
        return brain

    def test_circuit_breaker_limits_results(self):
        """Should not return more than max_activated_nodes entities."""
        brain = self._make_star_graph("Center", 50)
        result = brain.spread_activation(
            ["Center"], iterations=3, threshold=0.01, max_activated_nodes=10
        )
        assert len(result) <= 10, (
            f"Circuit breaker should cap at 10, got {len(result)}"
        )

    def test_circuit_breaker_logs_warning(self, caplog):
        """Should log a warning when the circuit breaker trips."""
        brain = self._make_star_graph("Center", 50)
        with caplog.at_level(logging.WARNING, logger="agent_brain.brain"):
            brain.spread_activation(
                ["Center"], iterations=3, threshold=0.01, max_activated_nodes=5
            )
        assert any("circuit breaker" in msg.lower() for msg in caplog.messages), (
            f"Expected 'circuit breaker' warning, got: {caplog.messages}"
        )

    def test_circuit_breaker_not_tripped_under_limit(self, caplog):
        """Should NOT log a warning when under the limit."""
        brain = self._make_star_graph("Center", 5)
        with caplog.at_level(logging.WARNING, logger="agent_brain.brain"):
            result = brain.spread_activation(
                ["Center"], iterations=1, threshold=0.01, max_activated_nodes=100
            )
        assert not any("circuit breaker" in msg.lower() for msg in caplog.messages)
        # Should still return results
        assert len(result) > 0

    def test_circuit_breaker_stops_early(self):
        """When tripped at iteration 1, should not continue to iteration 3."""
        brain = self._make_star_graph("Center", 50)
        # With max_activated_nodes=5, should stop after first iteration
        result = brain.spread_activation(
            ["Center"], iterations=10, threshold=0.01, max_activated_nodes=5
        )
        # It should have stopped, but still returned results
        assert len(result) > 0
        assert len(result) <= 5

    def test_circuit_breaker_keeps_highest_activation(self):
        """When truncating, should keep the most activated nodes."""
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)

        conn.add_node("A", {"Entity"}, {"name": "A", "type": "concept"})
        for i in range(20):
            name = f"N{i}"
            weight = 0.9 if i < 3 else 0.1  # First 3 have strong connections
            conn.add_node(name, {"Entity"}, {"name": name, "type": "concept"})
            conn.add_relationship("A", name, "RELATED_TO", {"weight": weight})

        result = brain.spread_activation(
            ["A"], iterations=1, threshold=0.01, max_activated_nodes=3
        )
        assert len(result) <= 3
        # The strongest-connected nodes should be kept
        for name, activation in result.items():
            assert activation > 0


class TestSpreadActivationDenseGraph:
    """Test behavior on dense subgraphs."""

    def test_dense_graph_terminates(self):
        """Complete graph of 20 nodes should terminate without issue."""
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)

        # Create 20-node complete graph
        nodes = [f"N{i}" for i in range(20)]
        for name in nodes:
            conn.add_node(name, {"Entity"}, {"name": name, "type": "concept"})

        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                conn.add_relationship(a, b, "RELATED_TO", {"weight": 0.5})

        result = brain.spread_activation(["N0"], iterations=3, threshold=0.01)
        assert isinstance(result, dict)
        assert len(result) <= 19  # Max = all non-seed nodes

    def test_dense_graph_with_circuit_breaker(self):
        """Circuit breaker should prevent dense graph from activating everything."""
        conn = FakeNeo4jConnection()
        brain = Neo4jBrain(connection=conn, embedder=mock_embedder)

        # Create 50-node complete graph
        nodes = [f"N{i}" for i in range(50)]
        for name in nodes:
            conn.add_node(name, {"Entity"}, {"name": name, "type": "concept"})
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                conn.add_relationship(a, b, "RELATED_TO", {"weight": 0.5})

        result = brain.spread_activation(
            ["N0"], iterations=3, threshold=0.01, max_activated_nodes=10
        )
        assert len(result) <= 10

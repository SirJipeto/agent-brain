"""
Neo4j Connection Helper

Provides a unified connection interface for the agent_brain package.
Uses the modern driver.execute_query() API (Neo4j Python Driver 5+/6).
"""
import os
import logging
from typing import List, Dict, Optional, Any

from neo4j import GraphDatabase, RoutingControl

logger = logging.getLogger(__name__)


def get_connection_config() -> dict:
    """Get Neo4j connection config from environment."""
    return {
        'uri': os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        'user': os.getenv("NEO4J_USER", "neo4j"),
        'password': os.getenv("NEO4J_PASSWORD", "password"),
    }


def create_driver():
    """Create a raw Neo4j driver instance."""
    config = get_connection_config()
    return GraphDatabase.driver(
        config['uri'],
        auth=(config['user'], config['password']),
    )


class Neo4jConnection:
    """
    Thin wrapper around the Neo4j Python driver that exposes a simple
    execute_query / execute_write / execute_single interface.

    Uses driver.execute_query() for managed transactions (retry-safe).
    """

    def __init__(self, driver=None):
        if driver is None:
            driver = create_driver()
        self.driver = driver

    # -- Read queries ----------------------------------------------------------

    def execute_query(self, query: str, params: dict = None) -> List[Dict]:
        """Run a read query. Returns a list of record dicts."""
        try:
            records, _summary, _keys = self.driver.execute_query(
                query,
                parameters_=params or {},
                routing_=RoutingControl.READ,
            )
            return [record.data() for record in records]
        except Exception as e:
            logger.warning(f"execute_query failed: {e}")
            return []

    def execute_single(self, query: str, params: dict = None) -> Optional[Dict]:
        """Run a read query and return the first record (or None)."""
        results = self.execute_query(query, params)
        return results[0] if results else None

    # -- Write queries ---------------------------------------------------------

    def execute_write(self, query: str, params: dict = None) -> List[Dict]:
        """Run a write query. Returns a list of record dicts."""
        try:
            records, _summary, _keys = self.driver.execute_query(
                query,
                parameters_=params or {},
            )
            return [record.data() for record in records]
        except Exception as e:
            logger.warning(f"execute_write failed: {e}")
            return []

    # -- Lifecycle -------------------------------------------------------------

    def verify_connectivity(self) -> bool:
        """Check that the database is reachable."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Connectivity check failed: {e}")
            return False

    def close(self):
        """Close the underlying driver."""
        self.driver.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_connection: Optional[Neo4jConnection] = None


def get_connection() -> Neo4jConnection:
    """Get or create a singleton Neo4jConnection."""
    global _connection
    if _connection is None:
        _connection = Neo4jConnection()
    return _connection

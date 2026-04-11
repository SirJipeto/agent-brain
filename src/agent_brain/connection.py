"""
Neo4j Connection Helper

Provides a unified connection interface for the agent_brain package.
Uses the modern driver.execute_query() API (Neo4j Python Driver 5+/6).
"""
import os
import logging
from typing import List, Dict, Optional, Any
import time

from neo4j import GraphDatabase, RoutingControl

logger = logging.getLogger(__name__)


class BrainConnectionError(Exception):
    """Raised when database operations fail persistently."""
    pass


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
    max_pool_size = int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "50"))
    return GraphDatabase.driver(
        config['uri'],
        auth=(config['user'], config['password']),
        max_connection_pool_size=max_pool_size
    )


class Neo4jConnection:
    """
    Thin wrapper around the Neo4j Python driver that exposes a simple
    execute_query / execute_write / execute_single interface.

    Uses driver.execute_query() for managed transactions (retry-safe).
    """

    def __init__(self, driver=None, max_retries: int = 2):
        if driver is None:
            driver = create_driver()
        self.driver = driver
        self.max_retries = max_retries

    # -- Read queries ----------------------------------------------------------

    def execute_query(self, query: str, params: dict = None) -> List[Dict]:
        """Run a read query. Returns a list of record dicts."""
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                records, _summary, _keys = self.driver.execute_query(
                    query,
                    parameters_=params or {},
                    routing_=RoutingControl.READ,
                )
                return [record.data() for record in records]
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    sleep_time = 0.5 * (2 ** attempt)
                    logger.warning(
                        f"execute_query failed (attempt {attempt+1}/{self.max_retries+1}), "
                        f"retrying in {sleep_time}s: {e}"
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(f"execute_query failed persistently after {self.max_retries+1} attempts: {e}")
        
        raise BrainConnectionError(f"Persistent failure executing read query: {last_exception}") from last_exception

    def execute_single(self, query: str, params: dict = None) -> Optional[Dict]:
        """Run a read query and return the first record (or None)."""
        results = self.execute_query(query, params)
        return results[0] if results else None

    # -- Write queries ---------------------------------------------------------

    def execute_write(self, query: str, params: dict = None) -> List[Dict]:
        """Run a write query. Returns a list of record dicts."""
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                records, _summary, _keys = self.driver.execute_query(
                    query,
                    parameters_=params or {},
                )
                return [record.data() for record in records]
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    sleep_time = 0.5 * (2 ** attempt)
                    logger.warning(
                        f"execute_write failed (attempt {attempt+1}/{self.max_retries+1}), "
                        f"retrying in {sleep_time}s: {e}"
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(f"execute_write failed persistently after {self.max_retries+1} attempts: {e}")
        
        raise BrainConnectionError(f"Persistent failure executing write query: {last_exception}") from last_exception

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


def reset_connection():
    """Reset the singleton connection, destroying the existing instance."""
    global _connection
    if _connection is not None:
        try:
            _connection.close()
        except Exception:
            pass
        _connection = None

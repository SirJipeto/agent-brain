"""
Integration test fixtures — real Neo4j connection.

These fixtures require a running Neo4j instance. Start one with:
    docker compose -f docker-compose.test.yml up -d

Run integration tests with:
    pytest tests/integration/ -m integration
"""

import os
import time
import pytest

from agent_brain.connection import Neo4jConnection
from agent_brain.brain import Neo4jBrain


def _wait_for_neo4j(uri: str, user: str, password: str, max_retries: int = 30):
    """Wait for Neo4j to become available."""
    from neo4j import GraphDatabase
    for i in range(max_retries):
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            driver.close()
            return True
        except Exception:
            time.sleep(1)
    return False


@pytest.fixture(scope="session")
def neo4j_config():
    """Neo4j connection config for integration tests."""
    return {
        'uri': os.getenv("NEO4J_TEST_URI", "bolt://localhost:7688"),
        'user': os.getenv("NEO4J_TEST_USER", "neo4j"),
        'password': os.getenv("NEO4J_TEST_PASSWORD", "testpassword"),
    }


@pytest.fixture(scope="session")
def neo4j_driver(neo4j_config):
    """
    Session-scoped Neo4j driver.
    Waits for the test container to be ready.
    """
    from neo4j import GraphDatabase

    uri = neo4j_config['uri']
    user = neo4j_config['user']
    password = neo4j_config['password']

    if not _wait_for_neo4j(uri, user, password, max_retries=30):
        pytest.skip("Neo4j test instance not available — start with: docker compose -f docker-compose.test.yml up -d")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    yield driver
    driver.close()


@pytest.fixture
def neo4j_conn(neo4j_driver):
    """
    Per-test Neo4j connection.
    Cleans database before each test for isolation.
    """
    conn = Neo4jConnection(driver=neo4j_driver)

    # Clean all data before each test
    conn.execute_write("MATCH (n) DETACH DELETE n")

    yield conn


@pytest.fixture
def integration_brain(neo4j_conn):
    """
    Per-test Neo4jBrain with real Neo4j connection.
    Uses mock embedder to avoid sentence-transformers dependency in CI.
    """
    from tests.conftest import mock_embedder
    brain = Neo4jBrain(connection=neo4j_conn, embedder=mock_embedder)
    return brain

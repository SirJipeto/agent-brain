"""
Shared test fixtures for agent_brain tests.

Provides:
- FakeNeo4jConnection: In-memory mock of Neo4jConnection (no Neo4j required)
- Mock embedder: Deterministic embeddings for testing
- Pre-built fixtures for common test scenarios
"""

import uuid
import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

import pytest


# ============================================================================
# FAKE NEO4J CONNECTION
# ============================================================================

class FakeNeo4jConnection:
    """
    In-memory mock of Neo4jConnection for unit testing.

    Stores nodes and relationships in Python dicts. Supports a subset of
    Cypher patterns via simple regex parsing — enough for the queries used
    in agent_brain modules.

    This is NOT a full Cypher engine. It handles the specific query patterns
    used in brain.py, consolidation.py, and observer.py.
    """

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}       # id -> node props
        self.node_labels: Dict[str, set] = {}             # id -> set of labels
        self.relationships: List[Dict[str, Any]] = []     # list of rel dicts
        self._call_log: List[Tuple[str, dict]] = []       # (query, params) log
        self._custom_handlers: Dict[str, Any] = {}        # pattern -> handler

    # -- Core interface (matches Neo4jConnection) ----------------------------

    def execute_query(self, query: str, params: dict = None) -> List[Dict]:
        """Run a read query. Returns a list of record dicts."""
        params = params or {}
        self._call_log.append(("read", query, params))
        return self._dispatch(query, params)

    def execute_single(self, query: str, params: dict = None) -> Optional[Dict]:
        """Run a read query and return the first record (or None)."""
        results = self.execute_query(query, params)
        return results[0] if results else None

    def execute_write(self, query: str, params: dict = None) -> List[Dict]:
        """Run a write query. Returns a list of record dicts."""
        params = params or {}
        self._call_log.append(("write", query, params))
        return self._dispatch(query, params)

    def verify_connectivity(self) -> bool:
        return True

    def close(self):
        pass

    # -- Convenience methods for test setup ----------------------------------

    def add_node(self, node_id: str, labels: set, props: dict):
        """Directly insert a node for test setup."""
        self.nodes[node_id] = {**props, 'id': node_id}
        self.node_labels[node_id] = labels

    def add_relationship(self, from_id: str, to_id: str, rel_type: str, props: dict = None):
        """Directly insert a relationship for test setup."""
        self.relationships.append({
            'from': from_id,
            'to': to_id,
            'type': rel_type,
            'props': props or {},
        })

    def get_nodes_by_label(self, label: str) -> List[Dict]:
        """Get all nodes with a given label."""
        return [
            self.nodes[nid]
            for nid, labels in self.node_labels.items()
            if label in labels and nid in self.nodes
        ]

    def get_relationships_between(self, from_id: str, to_id: str) -> List[Dict]:
        """Get relationships between two nodes."""
        return [
            r for r in self.relationships
            if r['from'] == from_id and r['to'] == to_id
        ]

    def get_call_log(self) -> List[Tuple[str, dict]]:
        """Return the full query log for assertions."""
        return self._call_log

    def clear(self):
        """Reset all data."""
        self.nodes.clear()
        self.node_labels.clear()
        self.relationships.clear()
        self._call_log.clear()

    # -- Query dispatch ------------------------------------------------------

    def _dispatch(self, query: str, params: dict) -> List[Dict]:
        """
        Route a Cypher query to the appropriate handler.
        Uses keyword matching to determine query intent.
        """
        q = query.strip().upper()

        # Schema operations (constraints, indexes) — no-op in mock
        if 'CREATE CONSTRAINT' in q or 'CREATE INDEX' in q or 'CREATE FULLTEXT' in q or 'CREATE VECTOR' in q:
            return []

        # CALL { ... } subquery pattern (used in get_stats)
        if q.startswith('CALL'):
            return self._handle_stats_query(query, params)

        # MERGE (upsert) — check before CREATE since MERGE queries may also contain CREATE
        if 'MERGE' in q:
            return self._handle_merge(query, params)

        # SET (update) — check before CREATE to avoid matching CREATED_AT property names
        if 'SET' in q and 'MATCH' in q:
            return self._handle_update(query, params)

        # DELETE
        if 'DELETE' in q:
            return self._handle_delete(query, params)

        # CREATE node — use 'CREATE (' to match node creation, not property names like CREATED_AT
        if 'CREATE (' in q or 'CREATE(' in q:
            return self._handle_create(query, params)

        # MATCH (read)
        if 'MATCH' in q:
            return self._handle_match(query, params)

        return []

    def _handle_create(self, query: str, params: dict) -> List[Dict]:
        """Handle CREATE node queries."""
        # Extract label from CREATE (x:Label {
        label_match = re.search(r'CREATE\s+\(\w+:(\w+)', query, re.I)
        label = label_match.group(1) if label_match else 'Unknown'

        node_id = params.get('id', str(uuid.uuid4()))
        props = {k: v for k, v in params.items()}
        props['id'] = node_id

        self.nodes[node_id] = props
        self.node_labels[node_id] = {label}

        return [{'id': node_id}]

    def _handle_merge(self, query: str, params: dict) -> List[Dict]:
        """Handle MERGE (upsert) queries."""
        # Try to find existing node by name or id
        name = params.get('name')
        node_id = params.get('entity_id') or params.get('id') or str(uuid.uuid4())

        # Extract label
        label_match = re.search(r'MERGE\s+\(\w+:(\w+)', query, re.I)
        label = label_match.group(1) if label_match else 'Entity'

        if name:
            # Find existing by name
            existing_id = None
            for nid, props in self.nodes.items():
                if props.get('name') == name and label in self.node_labels.get(nid, set()):
                    existing_id = nid
                    break

            if existing_id:
                # Update existing
                for k, v in params.items():
                    if k not in ('name',):  # Don't overwrite merge key
                        self.nodes[existing_id][k] = v
                node_id = existing_id
            else:
                # Create new
                props = {k: v for k, v in params.items()}
                props['id'] = node_id
                self.nodes[node_id] = props
                self.node_labels[node_id] = {label}

        # Handle relationship creation in MERGE queries
        if 'MERGE' in query and '-[' in query:
            rel_match = re.search(r'MERGE\s+\(\w+\)-\[[\w:]*:?(\w+)\]', query, re.I)
            if not rel_match:
                # Try more patterns
                rel_match = re.search(r'\[r:(\w+)\]', query, re.I)

            if rel_match:
                rel_type = rel_match.group(1)
                memory_id = params.get('memory_id', '')
                if memory_id and node_id:
                    self.add_relationship(memory_id, node_id, rel_type, {
                        'relevance': params.get('relevance', 0.5),
                        'confidence': params.get('confidence', 0.5),
                        'context': params.get('context', ''),
                        'updated_at': datetime.now().isoformat()
                    })
                elif params.get('a') and params.get('b'):
                    # relate_entities pattern
                    a_name = params.get('a')
                    b_name = params.get('b')
                    a_id = next((nid for nid, p in self.nodes.items() if p.get('name') == a_name), None)
                    b_id = next((nid for nid, p in self.nodes.items() if p.get('name') == b_name), None)
                    if a_id and b_id:
                        self.add_relationship(a_id, b_id, rel_type, {
                            'weight': params.get('weight', 0.3),
                            'confidence': params.get('confidence', 0.5),
                            'context': params.get('context', ''),
                            'updated_at': datetime.now().isoformat()
                        })

        return [{'entity_id': node_id}]

    def _handle_match(self, query: str, params: dict) -> List[Dict]:
        """Handle MATCH read queries."""
        q_upper = query.upper()

        # Memory -> Entity (MENTIONS) relationship query (by memory id)
        if 'MENTIONS' in q_upper and params.get('id'):
            results = []
            for rel in self.relationships:
                if rel['from'] == params['id'] and rel['type'] == 'MENTIONS':
                    entity = self.nodes.get(rel['to'])
                    if entity:
                        results.append({'name': entity.get('name', '')})
            return results

        # Entity <- Memory (reverse MENTIONS, get_related_memories)
        if 'MENTIONS' in q_upper and params.get('name'):
            results = []
            entity_id = None
            for nid, props in self.nodes.items():
                if props.get('name') == params['name']:
                    entity_id = nid
                    break
            if entity_id:
                for rel in self.relationships:
                    if rel['to'] == entity_id and rel['type'] == 'MENTIONS':
                        memory = self.nodes.get(rel['from'])
                        if memory:
                            results.append({
                                'id': memory.get('id', ''),
                                'content': memory.get('content', ''),
                                'summary': memory.get('summary', ''),
                                'timestamp': memory.get('timestamp', ''),
                                'importance': memory.get('importance', 0.5),
                            })
            top_k = params.get('top_k', 5)
            return results[:top_k]

        # Connected entities for activation spreading
        # Query pattern: MATCH (e:Entity {name: $name})-[r]-(connected) WHERE connected:Entity
        if params.get('name') and 'ENTITY' in q_upper and 'CONNECTED' in q_upper:
            return self._handle_connected(params['name'])

        # Graph traversal (simplified — returns direct connections)
        if params.get('seed'):
            return self._handle_traverse(params)

        # Memory by ID
        if 'MEMORY' in q_upper and params.get('id'):
            node = self.nodes.get(params['id'])
            if node:
                return [node]
            return []

        # Entity by name (simple lookup — NOT the connected-entities pattern)
        if 'ENTITY' in q_upper and params.get('name'):
            for nid, props in self.nodes.items():
                if props.get('name') == params['name'] and 'Entity' in self.node_labels.get(nid, set()):
                    return [props]
            return []

        # Memory search (semantic or text)
        if 'MEMORY' in q_upper and 'EMBEDDING' in q_upper:
            return self._handle_semantic_search(params)

        if 'MEMORY' in q_upper and 'ARCHIVED' in q_upper:
            return self._handle_memory_list(params)

        # Person query
        if 'PERSON' in q_upper:
            return self._handle_person_query(params)

        # Default: return all matching nodes
        return []

    def _handle_delete(self, query: str, params: dict) -> List[Dict]:
        """Handle DELETE queries."""
        threshold = params.get('threshold', 0.0)
        removed = []
        self.relationships = [
            r for r in self.relationships
            if not (r['props'].get('weight') is not None and r['props']['weight'] < threshold)
            or removed.append(r) is not None  # side-effect to count
        ]
        # Re-filter properly
        kept = []
        removed_count = 0
        for r in self.relationships:
            w = r['props'].get('weight')
            if w is not None and w < threshold:
                removed_count += 1
            else:
                kept.append(r)
        self.relationships = kept
        return [{'deleted': removed_count}]

    def _handle_update(self, query: str, params: dict) -> List[Dict]:
        """Handle SET/UPDATE queries."""
        q_upper = query.upper()

        # Archive old memories (MATCH + SET archived, has 'days' param)
        if 'ARCHIVED' in q_upper and params.get('days') is not None:
            count = 0
            for nid, labels in self.node_labels.items():
                if 'Memory' in labels:
                    node = self.nodes[nid]
                    if not node.get('archived', False):
                        node['archived'] = True
                        node['embedding'] = None
                        count += 1
            return [{'archived': count}]

        # Decay all weights
        if 'WEIGHT' in q_upper and 'factor' in str(params):
            factor = params.get('factor', 0.95)
            count = 0
            for r in self.relationships:
                if r['props'].get('weight') is not None:
                    r['props']['weight'] *= factor
                    count += 1
            return [{'decayed': count}]

        # Archive single memory by id
        if 'ARCHIVED' in q_upper and params.get('id'):
            node = self.nodes.get(params['id'])
            if node:
                node['archived'] = True
                node['embedding'] = None
            return []

        # Boost relationship
        if 'BOOST' in str(params):
            boost = params.get('boost', 0.1)
            a_name = params.get('a')
            b_name = params.get('b')
            for r in self.relationships:
                from_node = self.nodes.get(r['from'], {})
                to_node = self.nodes.get(r['to'], {})
                if (from_node.get('name') == a_name and to_node.get('name') == b_name) or \
                   (from_node.get('name') == b_name and to_node.get('name') == a_name):
                    w = r['props'].get('weight', 0.5)
                    r['props']['weight'] = min(1.0, w + boost)
            return []

        # Decay by age (has 'days' and 'factor' params)
        if params.get('days') is not None and params.get('factor') is not None:
            return [{'decayed': 0}]

        return []

    def _handle_stats_query(self, query: str, params: dict) -> List[Dict]:
        """Handle CALL { ... } subquery stats pattern."""
        memories = len([n for n, labels in self.node_labels.items()
                       if 'Memory' in labels and not self.nodes[n].get('archived', False)])
        entities = len([n for n, labels in self.node_labels.items() if 'Entity' in labels])
        relationships = len(self.relationships)
        return [{'memories': memories, 'entities': entities, 'relationships': relationships}]

    def _handle_traverse(self, params: dict) -> List[Dict]:
        """Simplified graph traversal for testing."""
        seed = params.get('seed', '')
        results = []

        # Find seed entity
        seed_id = None
        for nid, props in self.nodes.items():
            if props.get('name') == seed:
                seed_id = nid
                break

        if not seed_id:
            return []

        # Find direct connections
        for rel in self.relationships:
            connected_id = None
            if rel['from'] == seed_id:
                connected_id = rel['to']
            elif rel['to'] == seed_id:
                connected_id = rel['from']

            if connected_id and connected_id in self.nodes:
                connected = self.nodes[connected_id]
                
                # Calculate path weight matching real Cypher
                weight = rel['props'].get('weight', 0.3)
                confidence = rel['props'].get('confidence', 0.5)
                # Mock age decay: assume 0.8 if no updated_at, else simplified decay
                age_decay = 0.8
                updated_at = rel['props'].get('updated_at')
                if updated_at:
                    try:
                        dt = datetime.fromisoformat(updated_at)
                        days_old = (datetime.now() - dt).days
                        age_decay = 1.0 / (1.0 + 0.01 * max(0, days_old))
                    except:
                        pass
                
                path_weight = weight * confidence * age_decay
                
                results.append({
                    'path_nodes': [
                        {'name': seed, 'type': 'unknown'},
                        {'name': connected.get('name', ''), 'type': connected.get('type', 'unknown')}
                    ],
                    'relationship_types': [rel['type']],
                    'strength': path_weight,
                    'connected_name': connected.get('name', ''),
                    'connected_type': connected.get('type', 'unknown'),
                    'concept': connected.get('name', ''),  # for recall_associations
                    'type': connected.get('type', 'unknown'),
                    'depth': 1
                })

        # Sort by strength desc
        results.sort(key=lambda x: x['strength'], reverse=True)
        return results

    def _handle_connected(self, name: str) -> List[Dict]:
        """Get connected entities for activation spreading."""
        # Find node by name
        node_id = None
        for nid, props in self.nodes.items():
            if props.get('name') == name:
                node_id = nid
                break

        if not node_id:
            return []

        results = []
        for rel in self.relationships:
            connected_id = None
            if rel['from'] == node_id:
                connected_id = rel['to']
            elif rel['to'] == node_id:
                connected_id = rel['from']

            if connected_id and connected_id in self.nodes:
                connected = self.nodes[connected_id]
                if 'Entity' in self.node_labels.get(connected_id, set()):
                    results.append({
                        'name': connected.get('name', ''),
                        'connection_weight': rel['props'].get('weight', 0.5),
                    })

        return results

    def _handle_semantic_search(self, params: dict) -> List[Dict]:
        """Mock semantic search — returns memories sorted by importance."""
        top_k = params.get('top_k', 5)
        container = params.get('container')

        memories = []
        for nid, labels in self.node_labels.items():
            if 'Memory' not in labels:
                continue
            node = self.nodes[nid]
            if node.get('archived', False):
                continue
            if container and node.get('container') != container:
                continue
            if node.get('embedding') is None:
                continue
            memories.append({
                'id': node['id'],
                'content': node.get('content', ''),
                'summary': node.get('summary', ''),
                'timestamp': node.get('timestamp', ''),
                'importance': node.get('importance', 0.5),
                'tags': node.get('salience_tags', []),
                'score': node.get('importance', 0.5),  # Mock: use importance as score
            })

        memories.sort(key=lambda m: m['score'], reverse=True)
        return memories[:top_k]

    def _handle_memory_list(self, params: dict) -> List[Dict]:
        """List memories with filters."""
        results = []
        for nid, labels in self.node_labels.items():
            if 'Memory' not in labels:
                continue
            node = self.nodes[nid]
            if not node.get('archived', False):
                results.append({
                    'id': node['id'],
                    'content': node.get('content', ''),
                    'container': node.get('container', 'default'),
                    'source': node.get('source', ''),
                    'created_at': node.get('created_at', ''),
                })
        return results

    def _handle_person_query(self, params: dict) -> List[Dict]:
        """Handle Person node queries."""
        for nid, labels in self.node_labels.items():
            if 'Person' in labels:
                node = self.nodes[nid]
                return [{'goals': node.get('goals', [])}]
        return []


# ============================================================================
# MOCK EMBEDDER
# ============================================================================

def mock_embedder(text: str) -> List[float]:
    """
    Deterministic mock embedder for testing.
    Returns a 384-dim vector based on text hash.
    Same input always produces the same output.
    """
    import hashlib
    h = hashlib.sha256(text.encode()).hexdigest()
    vector = []
    for i in range(384):
        byte_val = int(h[(i * 2) % len(h)] + h[(i * 2 + 1) % len(h)], 16)
        vector.append((byte_val / 127.5) - 1.0)
    return vector


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def fake_conn():
    """Fresh FakeNeo4jConnection for each test."""
    return FakeNeo4jConnection()


@pytest.fixture
def brain(fake_conn):
    """Neo4jBrain instance with mock connection and mock embedder."""
    from agent_brain.brain import Neo4jBrain
    return Neo4jBrain(connection=fake_conn, embedder=mock_embedder)


@pytest.fixture
def brain_with_data(brain, fake_conn):
    """Neo4jBrain with pre-populated test data."""
    # Add some entities
    fake_conn.add_node("e1", {"Entity"}, {"name": "Python", "type": "technology", "importance": 0.8})
    fake_conn.add_node("e2", {"Entity"}, {"name": "Machine Learning", "type": "concept", "importance": 0.9})
    fake_conn.add_node("e3", {"Entity"}, {"name": "Neo4j", "type": "technology", "importance": 0.7})
    fake_conn.add_node("e4", {"Entity"}, {"name": "Alice", "type": "person", "importance": 0.6})
    fake_conn.add_node("e5", {"Entity"}, {"name": "Tokyo", "type": "place", "importance": 0.5})

    # Add relationships
    fake_conn.add_relationship("e1", "e2", "RELATED_TO", {"weight": 0.8})
    fake_conn.add_relationship("e2", "e3", "ENABLES", {"weight": 0.6})
    fake_conn.add_relationship("e3", "e1", "DEPENDS_ON", {"weight": 0.7})
    fake_conn.add_relationship("e4", "e5", "LOCATED_AT", {"weight": 0.9})

    # Add some memories
    fake_conn.add_node("m1", {"Memory"}, {
        "content": "We discussed Python and machine learning today",
        "summary": "Python/ML discussion",
        "content_type": "conversation",
        "embedding": mock_embedder("We discussed Python and machine learning today"),
        "importance": 0.7,
        "salience_tags": ["technology"],
        "container": "default",
        "timestamp": datetime.now().isoformat(),
        "source": "user_message",
        "archived": False,
    })
    fake_conn.add_node("m2", {"Memory"}, {
        "content": "Alice is traveling to Tokyo next month",
        "summary": "Alice Tokyo trip",
        "content_type": "conversation",
        "embedding": mock_embedder("Alice is traveling to Tokyo next month"),
        "importance": 0.6,
        "salience_tags": ["travel"],
        "container": "default",
        "timestamp": datetime.now().isoformat(),
        "source": "user_message",
        "archived": False,
    })

    # Link memories to entities
    fake_conn.add_relationship("m1", "e1", "MENTIONS", {"relevance": 0.9})
    fake_conn.add_relationship("m1", "e2", "MENTIONS", {"relevance": 0.8})
    fake_conn.add_relationship("m2", "e4", "MENTIONS", {"relevance": 0.9})
    fake_conn.add_relationship("m2", "e5", "MENTIONS", {"relevance": 0.8})

    # Add a Person node for user goals
    fake_conn.add_node("user", {"Person"}, {
        "name": "user",
        "goals": ["learn Rust", "launch startup"]
    })

    return brain


@pytest.fixture
def consolidator(brain_with_data):
    """MemoryConsolidator with pre-populated brain."""
    from agent_brain.consolidation import MemoryConsolidator
    return MemoryConsolidator(brain_with_data)


@pytest.fixture
def observer(brain_with_data):
    """ObserverFramework with pre-populated brain."""
    from agent_brain.observer import ObserverFramework
    return ObserverFramework(brain_with_data)

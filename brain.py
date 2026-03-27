"""
Neo4jBrain: Main GraphRAG implementation for associative memory
Implements hybrid vector + graph search, entity extraction, and proactive recall
"""

import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import logging

from .connection import get_connection, Neo4jConnection

logger = logging.getLogger(__name__)


@dataclass
class MemoryNode:
    """Represents a memory node in the graph"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    summary: str = ""
    content_type: str = "conversation"
    embedding: Optional[List[float]] = None
    importance: float = 0.5
    salience_tags: List[str] = field(default_factory=list)
    container: str = "default"
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'summary': self.summary,
            'content_type': self.content_type,
            'embedding': self.embedding,
            'importance': self.importance,
            'salience_tags': self.salience_tags,
            'container': self.container,
            'timestamp': str(self.timestamp),
            'source': self.source,
            'metadata': self.metadata
        }


@dataclass  
class EntityNode:
    """Represents an entity node in the graph"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entity_type: str = "concept"  # person, place, object, organization, concept, event, technology
    description: str = ""
    embedding: Optional[List[float]] = None
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.entity_type,
            'description': self.description,
            'embedding': self.embedding,
            'importance': self.importance,
            'tags': self.tags
        }


class Neo4jBrain:
    """
    Main brain class implementing GraphRAG for associative memory.

    Features:
    - Native vector search for semantic similarity
    - Graph traversal for multi-hop reasoning
    - Hybrid search combining vector + graph
    - Entity extraction and relationship building
    - Spontaneous recall via activation spreading
    """

    def __init__(self, connection: Optional[Neo4jConnection] = None, 
                 embedder=None):
        """
        Initialize the brain.

        Args:
            connection: Neo4j connection (creates new if None)
            embedder: Embedding function (uses mock if None)
        """
        self.conn = connection or get_connection()
        self._embedder = embedder or self._default_embedder
        self._embedding_model = None  # Lazy-loaded by _default_embedder
        self._ensure_schema()

    def _default_embedder(self, text: str) -> List[float]:
        """Real semantic embedder using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers not installed. Falling back to mock embedder.")
            import hashlib
            h = hashlib.sha256(text.encode()).hexdigest()
            # Build a 384-dim vector with values in [-1, 1]
            vector = []
            for i in range(384):
                byte_val = int(h[(i * 2) % len(h)] + h[(i * 2 + 1) % len(h)], 16)
                vector.append((byte_val / 127.5) - 1.0)
            return vector

        # Load embedding model on first use to save startup time
        if self._embedding_model is None:
            logger.info("Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        embedding = self._embedding_model.encode(text)
        return embedding.tolist()

    def _ensure_schema(self):
        """Ensure constraints and indexes exist"""
        constraints = [
            "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE"
        ]

        for c in constraints:
            try:
                self.conn.execute_write(c)
            except Exception as e:
                # Constraint may already exist
                logger.debug(f"Constraint check: {e}")

    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text"""
        return self._embedder(text)

    # ============================================
    # MEMORY OPERATIONS
    # ============================================

    def add_memory(self, content: str, summary: str = "",
                   content_type: str = "conversation",
                   importance: float = 0.5,
                   salience_tags: Optional[List[str]] = None,
                   container: str = "default",
                   source: str = "",
                   metadata: Optional[Dict] = None,
                   entities: Optional[List[Dict]] = None,
                   extract_entities: bool = True) -> str:
        """
        Add a memory to the graph with automatic entity extraction.

        Args:
            content: Full memory content
            summary: Brief summary
            content_type: conversation, document, extracted, reflection, note
            importance: 0.0-1.0 importance score
            salience_tags: Tags like ["goal", "preference", "deadline"]
            container: Context grouping
            source: Origin (e.g., "user_message", "document:/path")
            metadata: Additional metadata
            entities: Pre-extracted entities (optional)
            extract_entities: Whether to extract entities from content

        Returns:
            Memory ID
        """
        memory_id = str(uuid.uuid4())
        embedding = self._embed(content)
        timestamp = datetime.now()

        if salience_tags is None:
            salience_tags = []
        if metadata is None:
            metadata = {}

        # Store memory node
        query = """
        CREATE (m:Memory {
            id: $id,
            content: $content,
            summary: $summary,
            content_type: $content_type,
            embedding: $embedding,
            importance: $importance,
            salience_tags: $salience_tags,
            container: $container,
            timestamp: datetime($timestamp),
            source: $source,
            metadata: $metadata,
            created_at: datetime(),
            archived: false
        })
        RETURN m.id as id
        """

        self.conn.execute_write(query, {
            'id': memory_id,
            'content': content,
            'summary': summary or content[:200],
            'content_type': content_type,
            'embedding': embedding,
            'importance': importance,
            'salience_tags': salience_tags,
            'container': container,
            'timestamp': timestamp.isoformat(),
            'source': source,
            'metadata': json.dumps(metadata)
        })

        # Extract and link entities
        if entities is None and extract_entities:
            entities = self.extract_entities(content)

        if entities:
            self._link_memory_to_entities(memory_id, entities, content)

        logger.info(f"Added memory {memory_id} with {len(entities or [])} entities")
        return memory_id

    def _link_memory_to_entities(self, memory_id: str, entities: List[Dict], 
                                  content: str):
        """Link a memory to its entities, creating/updating entity nodes"""

        for i, entity in enumerate(entities):
            name = entity.get('name', '').strip()
            if not name:
                continue

            entity_type = entity.get('type', 'concept')
            description = entity.get('description', '')

            # Create or merge entity node
            entity_id = str(uuid.uuid4())

            entity_query = """
            MERGE (e:Entity {name: $name})
            SET e.id = COALESCE(e.id, $entity_id),
                e.type = COALESCE(e.type, $type),
                e.description = COALESCE(e.description, $description),
                e.importance = COALESCE(e.importance, 0.5),
                e.updated_at = datetime()
            WITH e

            // Link to memory
            MATCH (m:Memory {id: $memory_id})
            MERGE (m)-[r:MENTIONS]->(e)
            SET r.relevance = $relevance,
                r.context = $context

            RETURN e.id as entity_id
            """

            # Calculate relevance based on position in text
            relevance = max(0.3, 1.0 - (i * 0.1))
            context = "mentioned" if name.lower() in content.lower() else "extracted"

            try:
                self.conn.execute_write(entity_query, {
                    'name': name,
                    'entity_id': entity_id,
                    'type': entity_type,
                    'description': description,
                    'memory_id': memory_id,
                    'relevance': relevance,
                    'context': context
                })
            except Exception as e:
                logger.warning(f"Failed to link entity {name}: {e}")

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from text using simple NLP.
        Replace with LLM-based extraction for production.

        Returns list of entity dicts with name, type, description
        """
        # Simple extraction - for production use LLM-based extraction
        import re

        entities = []
        seen = set()

        # Capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for word in capitalized[:10]:  # Limit to 10
            if word.lower() not in seen and len(word) > 2:
                seen.add(word.lower())
                entities.append({
                    'name': word,
                    'type': 'concept',
                    'description': f"Extracted from text"
                })

        # Key phrases/quotes
        quotes = re.findall(r'"([^"]+)"', text)
        for quote in quotes[:3]:
            if quote.lower() not in seen and len(quote) > 3:
                seen.add(quote.lower())
                entities.append({
                    'name': quote,
                    'type': 'concept',
                    'description': "Quoted phrase"
                })

        return entities

    # ============================================
    # SEARCH OPERATIONS
    # ============================================

    def semantic_search(self, query_str: str, top_k: int = 5, 
                       container: Optional[str] = None) -> List[Dict]:
        """
        Vector similarity search for semantically similar memories.
        """
        query_embedding = self._embed(query_str)

        params = {
            'query_embedding': query_embedding,
            'top_k': top_k
        }

        container_clause = ""
        if container:
            container_clause = "AND m.container = $container"
            params['container'] = container

        cypher = f"""
        MATCH (m:Memory)
        WHERE m.embedding IS NOT NULL AND m.archived = false {container_clause}
        WITH m,
             reduce(dot = 0.0, i IN range(0, size(m.embedding)-1) |
                    dot + m.embedding[i] * $query_embedding[i]) /
             (sqrt(reduce(a = 0.0, i IN range(0, size(m.embedding)-1) |
                    a + m.embedding[i]^2)) *
              sqrt(reduce(b = 0.0, i IN range(0, size($query_embedding)-1) |
                    b + $query_embedding[i]^2)) + 1e-10) AS score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN m.id as id, m.content as content, m.summary as summary,
               m.timestamp as timestamp, m.importance as importance,
               m.salience_tags as tags, score
        """

        try:
            results = self.conn.execute_query(cypher, params)
            return [dict(r) if isinstance(r, dict) else r for r in results]
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            # Fallback to text search
            return self._text_search_fallback(query_str, top_k, container)

    def _text_search_fallback(self, query_str: str, top_k: int = 5,
                              container: Optional[str] = None) -> List[Dict]:
        """Fallback text-based search when vector search is unavailable."""
        import re as _re
        escaped = _re.escape(query_str)
        params = {
            'query': f"(?i).*{escaped}.*",
            'top_k': top_k,
        }

        container_clause = ""
        if container:
            container_clause = "AND m.container = $container"
            params['container'] = container

        cypher = f"""
        MATCH (m:Memory)
        WHERE m.content =~ $query AND m.archived = false {container_clause}
        RETURN m.id as id, m.content as content, m.summary as summary,
               m.timestamp as timestamp, m.importance as importance,
               m.salience_tags as tags, 1.0 as score
        ORDER BY m.importance DESC, m.timestamp DESC
        LIMIT $top_k
        """

        return self.conn.execute_query(cypher, params)

    def graph_traverse(self, seed_entity: str, depth: int = 2,
                      relationship_type: Optional[str] = None) -> List[Dict]:
        """
        Traverse the graph from a seed entity.
        Multi-hop reasoning to find connected concepts.
        """
        rel_clause = ""
        if relationship_type:
            rel_clause = f":{relationship_type}"

        query = f"""
        MATCH path = (start:Entity {{name: $seed}})-[{rel_clause}*1..{depth}]-(connected)
        WHERE start <> connected
        WITH path, connected,
             reduce(weight = 1.0, r IN relationships(path) | 
                   weight * COALESCE(r.weight, 0.5)) as path_weight
        RETURN 
            [n IN nodes(path) | {{name: n.name, type: COALESCE(n.type, "unknown")}}] as path_nodes,
            [r IN relationships(path) | type(r)] as relationship_types,
            path_weight as strength,
            connected.name as connected_name,
            connected.type as connected_type
        ORDER BY strength DESC
        LIMIT 20
        """

        results = self.conn.execute_query(query, {'seed': seed_entity})
        return [dict(r) for r in results]

    def hybrid_search(self, query_str: str, top_k: int = 5,
                     container: Optional[str] = None) -> Dict:
        """
        Combine vector search with graph traversal for rich results.
        This is the core GraphRAG pattern.
        """
        # 1. Get semantic matches
        semantic_results = self.semantic_search(query_str, top_k * 2, container)

        # 2. Extract query entities
        query_entities = self.extract_entities(query_str)

        # 3. Traverse graph from query entities
        graph_insights = []
        for entity in query_entities[:3]:  # Limit to top 3 entities
            traversed = self.graph_traverse(entity['name'], depth=2)
            graph_insights.extend(traversed)

        # 4. Find related entities
        related_entities = list(set([
            r['connected_name'] for r in graph_insights 
            if r.get('connected_name')
        ]))

        # 5. Build enriched context
        return {
            'query': query_str,
            'semantic_matches': semantic_results[:top_k],
            'graph_insights': graph_insights[:10],
            'related_entities': related_entities[:10],
            'timestamp': datetime.now().isoformat()
        }

    # ============================================
    # ASSOCIATIVE OPERATIONS
    # ============================================

    def recall_associations(self, concept: str, depth: int = 3) -> List[Dict]:
        """
        Human-like recall: A → B → C chains.
        Find all concepts connected through chains.
        """
        query = f"""
        MATCH path = (start:Entity)-[:RELATED_TO|KNOWS_ABOUT*1..{depth}]-(end)
        WHERE start.name = $concept
        WITH path, end,
             reduce(w = 1.0, r IN relationships(path) | 
                   w * COALESCE(r.weight, 0.5)) as path_weight
        RETURN 
            end.name as concept,
            end.type as type,
            path_weight as strength,
            [n IN nodes(path)[1..-1] | n.name] as intermediate,
            length(path) as depth
        ORDER BY strength DESC
        LIMIT 15
        """

        return [dict(r) for r in self.conn.execute_query(query, {'concept': concept})]

    def spread_activation(self, seed_entities: List[str], 
                         iterations: int = 3,
                         threshold: float = 0.1) -> Dict[str, float]:
        """
        Brain-like activation spreading from seed entities.

        Args:
            seed_entities: Starting nodes
            iterations: How many hops to spread
            threshold: Minimum activation to return

        Returns:
            Dict of entity -> activation level
        """
        activation = {e: 1.0 for e in seed_entities}

        for _ in range(iterations):
            new_activation = activation.copy()

            for entity, current in activation.items():
                if current < threshold:
                    continue

                # Get connections
                query = """
                MATCH (e:Entity {name: $name})-[r]-(connected)
                WHERE connected:Entity
                RETURN connected.name as name, 
                       COALESCE(r.weight, 0.5) as connection_weight
                """

                for record in self.conn.execute_query(query, {'name': entity}):
                    spread = current * record['connection_weight'] * 0.5
                    name = record['name']
                    new_activation[name] = new_activation.get(name, 0) + spread

            activation = new_activation

        # Filter and sort
        return {
            k: v for k, v in sorted(activation.items(), key=lambda x: -x[1])
            if v >= threshold and k not in seed_entities
        }

    def find_implicit_connections(self, entity_a: str, entity_b: str) -> List[Dict]:
        """
        Find hidden paths between two entities.
        "This relates to X through..."
        """
        cypher = """
        MATCH path = shortestPath(
            (a:Entity {name: $a})-[*1..4]-(b:Entity {name: $b})
        )
        WHERE a <> b
        WITH path
        WHERE ALL(r IN relationships(path) WHERE COALESCE(r.weight, 0) >= 0.2)
        RETURN 
            [n IN nodes(path) | n.name] as path,
            [r IN relationships(path) | type(r)] as edges,
            reduce(w = 1.0, r IN relationships(path) | 
                  w * COALESCE(r.weight, 0.5)) as path_strength
        ORDER BY path_strength DESC
        LIMIT 5
        """

        return self.conn.execute_query(cypher, {'a': entity_a, 'b': entity_b})

    def get_related_memories(self, entity_name: str, top_k: int = 5) -> List[Dict]:
        """Get memories related to an entity"""
        query = """
        MATCH (m:Memory)-[:MENTIONS]->(e:Entity {name: $name})
        RETURN m.id as id, m.content as content, m.summary as summary,
               m.timestamp as timestamp, m.importance as importance
        ORDER BY m.importance DESC, m.timestamp DESC
        LIMIT $top_k
        """

        return [dict(r) for r in self.conn.execute_query(query, 
                                                          {'name': entity_name, 'top_k': top_k})]

    # ============================================
    # RELATIONSHIP OPERATIONS
    # ============================================

    # Allowed relationship types (whitelist to prevent Cypher injection)
    ALLOWED_RELATIONSHIPS = {
        "RELATED_TO", "KNOWS_ABOUT", "INTERESTED_IN", "WORKS_ON",
        "PART_OF", "CAUSED_BY", "ENABLES", "DEPENDS_ON",
        "LOCATED_AT", "NEAR", "IS_A", "IMPLIES", "CONTRADICTS",
        "PREREQUISITE_FOR", "LEADS_TO", "MENTIONS",
    }

    def relate_entities(self, entity_a: str, entity_b: str,
                        relationship: str = "RELATED_TO",
                        weight: float = 0.5,
                        context: str = ""):
        """Create or update relationship between entities."""
        relationship = relationship.upper()
        if relationship not in self.ALLOWED_RELATIONSHIPS:
            logger.warning(f"Blocked disallowed relationship type: {relationship}")
            return

        cypher = f"""
        MERGE (a:Entity {{name: $a}})
        MERGE (b:Entity {{name: $b}})
        MERGE (a)-[r:{relationship}]->(b)
        SET r.weight = COALESCE(r.weight, 0.5) + $weight / 2,
            r.context = $context,
            r.updated_at = datetime()
        """

        self.conn.execute_write(cypher, {
            'a': entity_a,
            'b': entity_b,
            'weight': weight,
            'context': context,
        })

    def strengthen_relationship(self, entity_a: str, entity_b: str,
                                boost: float = 0.1):
        """Strengthen a relationship when entities co-occur"""
        query = """
        MATCH (a:Entity {name: $a})-[r]-(b:Entity {name: $b})
        SET r.weight = CASE WHEN r.weight + $boost > 1.0 THEN 1.0 ELSE r.weight + $boost END,
            r.last_cooccurred = datetime()
        """

        self.conn.execute_write(query, {'a': entity_a, 'b': entity_b, 'boost': boost})

    # ============================================
    # MAINTENANCE
    # ============================================

    def decay_weak_connections(self, threshold: float = 0.1):
        """Decay and prune weak relationships"""
        # Decay all weights (directed to avoid processing each rel twice)
        decay_query = """
        MATCH ()-[r]->()
        WHERE r.weight IS NOT NULL
        SET r.weight = r.weight * 0.95
        """
        self.conn.execute_write(decay_query)

        # Remove very weak
        prune_query = """
        MATCH ()-[r]->()
        WHERE r.weight IS NOT NULL AND r.weight < $threshold
        DELETE r
        """
        self.conn.execute_write(prune_query, {'threshold': threshold})

    def archive_old_memories(self, older_than_days: int = 30):
        """Archive old memories, keeping entity links."""
        cypher = """
        MATCH (m:Memory)
        WHERE m.created_at < datetime() - duration({days: $days})
          AND m.archived = false
        SET m.archived = true,
            m.archive_date = date(),
            m.embedding = null
        RETURN count(m) as archived
        """

        result = self.conn.execute_write(cypher, {'days': older_than_days})
        return result[0]['archived'] if result else 0

    def get_stats(self) -> Dict:
        """Get brain statistics"""
        query = """
        CALL { MATCH (m:Memory) WHERE m.archived = false RETURN count(m) AS memories }
        CALL { MATCH (e:Entity) RETURN count(e) AS entities }
        CALL { MATCH ()-[r]->() RETURN count(r) AS relationships }
        RETURN memories, entities, relationships
        """

        result = self.conn.execute_single(query)
        return dict(result) if result else {'memories': 0, 'entities': 0, 'relationships': 0}

    def get_user_goals(self) -> List[str]:
        """Retrieve user goals from the :Person node (if it exists)."""
        result = self.conn.execute_single("""
            MATCH (p:Person {id: "user"})
            RETURN p.goals AS goals
        """)
        if result and result.get('goals'):
            return result['goals']
        return []


# Convenience function
def get_brain(connection: Neo4jConnection = None) -> Neo4jBrain:
    """Get or create a Neo4jBrain instance"""
    return Neo4jBrain(connection)

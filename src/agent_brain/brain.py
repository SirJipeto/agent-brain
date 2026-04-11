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
from .metrics import trace_and_measure

logger = logging.getLogger(__name__)


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
                 embedder=None,
                 preload_embeddings: bool = True):
        """
        Initialize the brain.

        Args:
            connection: Neo4j connection (creates new if None)
            embedder: An EmbeddingProvider instance, a callable (text -> List[float]),
                      or None to auto-detect from environment.
            preload_embeddings: If True, warms up the embedding provider immediately
                                to avoid latency spikes on first use.

        Raises:
            embeddings.ProviderNotAvailableError: If no provider is configured
                and the default (sentence-transformers) is not installed.
        """
        from .embeddings import (
            EmbeddingProvider, CallableProvider,
            create_provider_from_env,
        )

        self.conn = connection or get_connection()

        # Resolve embedder: provider > callable > env auto-detect
        if embedder is None:
            self._provider = create_provider_from_env()
        elif isinstance(embedder, EmbeddingProvider):
            self._provider = embedder
        elif callable(embedder):
            # Backward compat: wrap raw callable
            self._provider = CallableProvider(embedder)
        else:
            raise TypeError(
                f"embedder must be an EmbeddingProvider, callable, or None. "
                f"Got: {type(embedder)}"
            )

        self.allowed_relationships = self.ALLOWED_RELATIONSHIPS.copy()
        self._ensure_schema()
        self._load_custom_relationships()

        if preload_embeddings:
            self.warmup()

    def warmup(self):
        """
        Explicitly pre-load the embedding model to avoid latency blocks on first use.
        Logs the time taken to load for diagnostics.
        """
        import time
        start_time = time.time()
        logger.info("Warming up embedding provider...")
        
        try:
            # Triggering a dummy embed forces lazy initialization
            self._embed("warmup")
            elapsed = time.time() - start_time
            logger.info(f"Embedding provider warmed up successfully in {elapsed:.3f}s")
        except Exception as e:
            logger.warning(f"Failed to warmup embedding provider: {e}")

    def _load_custom_relationships(self):
        """Load discovered custom relationship types from Neo4j config."""
        query = """
        MATCH (c:Config {key: "custom_relationship_types"})
        RETURN c.values as values
        """
        try:
            result = self.conn.execute_single(query)
            if result and result.get('values'):
                self.allowed_relationships.update(result['values'])
        except Exception as e:
            logger.warning(f"Failed to load custom relationship types: {e}")

    def get_health(self) -> dict:
        """Returns database connectivity and basic graph sizing metrics."""
        is_connected = self.conn.verify_connectivity()
        stats = {}
        if is_connected:
            try:
                stats = self.conn.execute_single("""
                CALL { MATCH (m:Memory) RETURN count(m) AS total_memories }
                CALL { MATCH (e:Entity) RETURN count(e) AS total_entities }
                CALL { MATCH ()-[r]->() RETURN count(r) AS total_relationships }
                RETURN total_memories, total_entities, total_relationships
                """)
            except Exception:
                stats = {}
        
        return {
            "status": "up" if is_connected else "down",
            "metrics": stats or {}
        }

    def _ensure_schema(self):
        """Ensure constraints and indexes exist"""
        constraints = [
            "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE FULLTEXT INDEX memory_content_fulltext IF NOT EXISTS FOR (m:Memory) ON EACH [m.content]"
        ]

        for c in constraints:
            try:
                self.conn.execute_write(c)
            except Exception as e:
                # Constraint may already exist
                logger.debug(f"Constraint check: {e}")

    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text using the configured provider."""
        return self._provider.embed(text)

    # ============================================
    # MEMORY OPERATIONS
    # ============================================

    @trace_and_measure("add_memory", is_memory_op=True)
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

    @trace_and_measure("read_memory")
    def read_memory(self, memory_id: str) -> Optional[Dict]:
        """Retrieve a specific memory by ID"""
        query = "MATCH (m:Memory {id: $id}) RETURN m"
        result = self.conn.execute_single(query, {'id': memory_id})
        return dict(result) if result else None

    @trace_and_measure("archive_memory", is_memory_op=True)
    def archive_memory(self, memory_id: str):
        """Mark a memory as archived (hidden from standard search)"""
        query = "MATCH (m:Memory {id: $id}) SET m.archived = true"
        self.conn.execute_write(query, {'id': memory_id})

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
        Extract entities from text using spaCy NER.

        Uses spaCy for named entity recognition with proper type classification
        (person, organization, place, date, event, etc.). Falls back to regex
        extraction if spaCy is not available.

        Returns list of entity dicts with name, type, description, confidence.
        """
        from .nlp import extract_entities_spacy
        return extract_entities_spacy(text)

    # ============================================
    # SEARCH OPERATIONS
    # ============================================

    @trace_and_measure("semantic_search")
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
            logger.warning(f"Vector search failed ({e}), retrying once...")
            try:
                import time
                time.sleep(1.0)
                results = self.conn.execute_query(cypher, params)
                return [dict(r) if isinstance(r, dict) else r for r in results]
            except Exception as e2:
                logger.warning(f"Vector search failed again: {e2}")
                # Fallback to text search
                return self._text_search_fallback(query_str, top_k, container)

    def _text_search_fallback(self, query_str: str, top_k: int = 5,
                              container: Optional[str] = None) -> List[Dict]:
        """Fallback text-based search when vector search is unavailable."""
        params = {
            'query': query_str,
            'top_k': top_k,
        }

        container_clause = ""
        if container:
            container_clause = "AND m.container = $container"
            params['container'] = container

        cypher = f"""
        CALL db.index.fulltext.queryNodes("memory_content_fulltext", $query) YIELD node AS m, score
        WHERE m.archived = false {container_clause}
        RETURN m.id as id, m.content as content, m.summary as summary,
               m.timestamp as timestamp, m.importance as importance,
               m.salience_tags as tags, score
        ORDER BY score DESC, m.timestamp DESC
        LIMIT $top_k
        """

        return self.conn.execute_query(cypher, params)

    @trace_and_measure("graph_traverse")
    def graph_traverse(self, seed_entity: str, depth: int = 2,
                      relationship_type: Optional[str] = None) -> List[Dict]:
        """
        Traverse the graph from a seed entity.
        Multi-hop reasoning to find connected concepts.

        Path strength is weighted by:
        - Relationship weight (explicit strength)
        - Relationship confidence (how sure we are about the link)
        - Age decay (older relationships score lower)
        """
        rel_clause = ""
        if relationship_type:
            rel_clause = f":{relationship_type}"

        query = f"""
        MATCH path = (start:Entity {{name: $seed}})-[{rel_clause}*1..{depth}]-(connected)
        WHERE start <> connected
        WITH path, connected,
             reduce(w = 1.0, r IN relationships(path) |
                   w * COALESCE(r.weight, 0.3)
                     * COALESCE(r.confidence, 0.5)
                     * CASE
                         WHEN r.updated_at IS NOT NULL
                         THEN 1.0 / (1.0 + 0.01 * duration.inDays(
                              r.updated_at, datetime()).days)
                         ELSE 0.8
                       END
             ) as path_weight
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

    @trace_and_measure("hybrid_search")
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

    @trace_and_measure("recall_associations")
    def recall_associations(self, concept: str, depth: int = 3) -> List[Dict]:
        """
        Human-like recall: A -> B -> C chains.
        Find all concepts connected through chains.

        Path weight incorporates relationship weight, confidence, and age decay.
        """
        query = f"""
        MATCH path = (start:Entity)-[:RELATED_TO|KNOWS_ABOUT*1..{depth}]-(end)
        WHERE start.name = $concept
        WITH path, end,
             reduce(w = 1.0, r IN relationships(path) |
                   w * COALESCE(r.weight, 0.3)
                     * COALESCE(r.confidence, 0.5)
                     * CASE
                         WHEN r.updated_at IS NOT NULL
                         THEN 1.0 / (1.0 + 0.01 * duration.inDays(
                              r.updated_at, datetime()).days)
                         ELSE 0.8
                       END
             ) as path_weight
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

    @trace_and_measure("spread_activation")
    def spread_activation(self, seed_entities: List[str],
                         iterations: int = 3,
                         threshold: float = 0.1,
                         decay_factor: float = 0.5,
                         max_activated_nodes: int = 100) -> Dict[str, float]:
        """
        Brain-like activation spreading from seed entities.

        Activation flows outward from seed nodes through weighted relationships,
        decaying at each hop. A circuit breaker stops spreading if too many
        nodes are activated (prevents runaway in dense subgraphs).

        Args:
            seed_entities: Starting nodes (activation = 1.0)
            iterations: How many hops to spread (default 3)
            threshold: Minimum activation to include in results (default 0.1)
            decay_factor: Multiplier applied at each hop (default 0.5).
                          Lower = faster decay, fewer activated nodes.
            max_activated_nodes: Circuit breaker — stop spreading if this many
                                 non-seed nodes are activated (default 100).

        Returns:
            Dict of entity name -> activation level, sorted descending.
        """
        activation = {e: 1.0 for e in seed_entities}
        seed_set = set(seed_entities)
        tripped = False

        for i in range(iterations):
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
                    spread = current * record['connection_weight'] * decay_factor
                    name = record['name']
                    new_activation[name] = new_activation.get(name, 0) + spread

            activation = new_activation

            # Circuit breaker: count non-seed activated nodes
            non_seed_count = sum(
                1 for k, v in activation.items()
                if k not in seed_set and v >= threshold
            )
            if non_seed_count >= max_activated_nodes:
                logger.warning(
                    f"Activation circuit breaker tripped: {non_seed_count} nodes "
                    f"activated (limit={max_activated_nodes}) after {i + 1} iterations. "
                    f"Stopping early to prevent runaway spreading."
                )
                tripped = True
                break

        # Filter and sort
        result = {
            k: v for k, v in sorted(activation.items(), key=lambda x: -x[1])
            if v >= threshold and k not in seed_set
        }

        # If circuit breaker tripped, truncate to max_activated_nodes
        if tripped and len(result) > max_activated_nodes:
            items = sorted(result.items(), key=lambda x: -x[1])[:max_activated_nodes]
            result = dict(items)

        return result

    @trace_and_measure("find_implicit_connections")
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

    @trace_and_measure("get_related_memories")
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

    # ALLOWED_RELATIONSHIPS is the base set of relationships.
    # self.allowed_relationships contains these plus any discovered at runtime.
    ALLOWED_RELATIONSHIPS = {
        "RELATED_TO", "KNOWS_ABOUT", "INTERESTED_IN", "WORKS_ON",
        "PART_OF", "CAUSED_BY", "ENABLES", "DEPENDS_ON",
        "LOCATED_AT", "NEAR", "IS_A", "IMPLIES", "CONTRADICTS",
        "PREREQUISITE_FOR", "LEADS_TO", "MENTIONS",
    }

    def register_relationship_type(self, rel_type: str):
        """Register a new custom relationship type and persist it."""
        rel_type = rel_type.upper().replace(" ", "_")
        if rel_type and rel_type not in self.allowed_relationships:
            self.allowed_relationships.add(rel_type)
            query = """
            MERGE (c:Config {key: "custom_relationship_types"})
            SET c.values = $values
            """
            custom_types = list(self.allowed_relationships - self.ALLOWED_RELATIONSHIPS)
            try:
                self.conn.execute_write(query, {'values': custom_types})
            except Exception as e:
                logger.warning(f"Failed to persist custom relationship type {rel_type}: {e}")

    @trace_and_measure("relate_entities")
    def relate_entities(self, entity_a: str, entity_b: str,
                        relationship: str = "RELATED_TO",
                        weight: float = 0.3,
                        confidence: float = 0.5,
                        context: str = "",
                        auto_discover: bool = False):
        """
        Create or update relationship between entities.

        Args:
            entity_a: Source entity name
            entity_b: Target entity name
            relationship: Relationship type
            weight: Relationship strength 0.0-1.0 (default 0.3)
            confidence: How certain we are about this link 0.0-1.0 (default 0.5)
            context: Optional context string describing why the link exists
            auto_discover: If True, dynamically add unknown relationship types
        """
        relationship = relationship.upper().replace(" ", "_")
        if relationship not in self.allowed_relationships:
            if auto_discover:
                logger.info(f"Auto-discovering new relationship type: {relationship}")
                self.register_relationship_type(relationship)
            else:
                logger.warning(f"Blocked disallowed relationship type: {relationship}")
                return

        cypher = f"""
        MERGE (a:Entity {{name: $a}})
        MERGE (b:Entity {{name: $b}})
        MERGE (a)-[r:{relationship}]->(b)
        SET r.weight = CASE
                WHEN r.weight IS NULL THEN $weight
                ELSE CASE WHEN r.weight + $weight / 2 > 1.0
                     THEN 1.0 ELSE r.weight + $weight / 2 END
            END,
            r.confidence = CASE
                WHEN r.confidence IS NULL THEN $confidence
                ELSE (r.confidence + $confidence) / 2.0
            END,
            r.context = $context,
            r.updated_at = datetime()
        """

        self.conn.execute_write(cypher, {
            'a': entity_a,
            'b': entity_b,
            'weight': weight,
            'confidence': confidence,
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

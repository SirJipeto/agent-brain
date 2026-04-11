"""
Memory Consolidation Manager

Handles periodic summarization of old memories to:
1. Prevent context rot (unbounded memory growth)
2. Extract persistent facts from episodic memories
3. Maintain lean, intelligent graph structure
4. Archive original memories while keeping entity links
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from .brain import Neo4jBrain

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation"""
    memories_processed: int
    memories_archived: int
    memories_consolidated: int
    facts_extracted: int
    new_summaries: List[str]
    timestamp: datetime


class MemoryConsolidator:
    """
    Periodically consolidates old memories into higher-level summaries.

    Process:
    1. Find old episodic memories (older than threshold)
    2. Group by topic/entity
    3. Generate summary for each cluster
    4. Extract persistent facts
    5. Archive originals (keep entity links)
    """

    def __init__(self, brain: Neo4jBrain, llm_summarizer: callable = None):
        """
        Args:
            brain: Neo4jBrain instance
            llm_summarizer: Optional function(content_list) -> summary string
        """
        self.brain = brain
        self._llm_summarizer = llm_summarizer or self._simple_summarizer

    def _simple_summarizer(self, memories: List[Dict]) -> str:
        """
        Simple summarizer without LLM.
        Extracts key entities and creates a summary.
        """
        if not memories:
            return ""

        # Extract all entities
        all_entities = set()
        contents = []

        for m in memories:
            content = m.get('content', '')
            if content:
                contents.append(content)

            # Get entities linked to this memory
            entity_results = self.brain.conn.execute_query("""
                MATCH (m:Memory {id: $id})-[:MENTIONS]->(e:Entity)
                RETURN e.name as name
            """, {'id': m.get('id', '')})
            for e in entity_results:
                if e.get('name'):
                    all_entities.add(e['name'])

        # Create simple summary
        topic = list(all_entities)[:3] if all_entities else ["various topics"]
        summary = f"Discussion involving {', '.join(topic)}. "
        summary += f"{len(memories)} exchanges covering key points."

        return summary

    def consolidate_old_memories(self, 
                                older_than_days: int = 7,
                                min_cluster_size: int = 2,
                                importance_threshold: float = 0.8,
                                dry_run: bool = False) -> ConsolidationResult:
        """
        Main consolidation entry point.

        Args:
            older_than_days: Consolidate memories older than this
            min_cluster_size: Minimum memories to form a cluster
            importance_threshold: Memories >= this importance are skipped
            dry_run: If True, only report what would be done

        Returns:
            ConsolidationResult with statistics
        """
        result = ConsolidationResult(
            memories_processed=0,
            memories_archived=0,
            memories_consolidated=0,
            facts_extracted=0,
            new_summaries=[],
            timestamp=datetime.now()
        )

        # 1. Find old memories
        old_memories = self._find_old_memories(older_than_days, importance_threshold)
        result.memories_processed = len(old_memories)

        if not old_memories:
            logger.info("No memories to consolidate")
            return result

        # 2. Group by topic/entity
        clusters = self._cluster_by_topic(old_memories)

        for topic, cluster_memories in clusters.items():
            if len(cluster_memories) < min_cluster_size:
                continue

            # 3. Generate summary
            summary = self._llm_summarizer(cluster_memories)

            if not summary:
                continue

            # 4. Extract facts from cluster
            facts = self._extract_facts_from_cluster(cluster_memories)
            result.facts_extracted += len(facts)

            if dry_run:
                result.new_summaries.append(f"[DRY RUN] Would consolidate {len(cluster_memories)} memories into: {summary[:50]}...")
                continue

            # 5. Store consolidated memory
            consolidated_id = self._store_consolidated_memory(
                summary, 
                cluster_memories,
                topic
            )

            # 6. Store extracted facts
            for fact in facts:
                self._store_fact(fact)

            # 7. Archive originals
            for m in cluster_memories:
                self._archive_memory(m['id'])
                result.memories_archived += 1

            result.memories_consolidated += 1
            result.new_summaries.append(f"Consolidated {len(cluster_memories)} memories → {summary[:50]}...")

        logger.info(f"Consolidation complete: {result}")
        return result

    def _find_old_memories(self, older_than_days: int, importance_threshold: float = 0.8) -> List[Dict]:
        """Find memories older than threshold"""
        query = """
        MATCH (m:Memory)
        WHERE m.archived = false
          AND m.created_at < datetime() - duration({days: $days})
          AND m.content_type IN ["conversation", "note"]
          AND coalesce(m.importance, 0.5) < $importance_threshold
        RETURN m.id as id, m.content as content, 
               m.container as container, m.source as source,
               m.created_at as created_at,
               m.metadata as metadata
        ORDER BY m.created_at ASC
        """

        results = self.brain.conn.execute_query(query, {
            'days': older_than_days,
            'importance_threshold': importance_threshold
        })
        return [dict(r) for r in results]

    def _cluster_by_topic(self, memories: List[Dict]) -> Dict[str, List[Dict]]:
        """Group memories by shared entities"""
        clusters = {}
        memory_entities = {}

        # First pass: get entities for each memory
        for m in memories:
            entity_results = self.brain.conn.execute_query("""
                MATCH (m:Memory {id: $id})-[:MENTIONS]->(e:Entity)
                RETURN e.name as name
            """, {'id': m['id']})
            entities = [r.get('name') for r in entity_results if r.get('name')]
            memory_entities[m['id']] = entities

        # Group by container (conversation context)
        container_clusters = {}
        for m in memories:
            container = m.get('container', 'default')
            if container not in container_clusters:
                container_clusters[container] = []
            container_clusters[container].append(m)

        # Also group by shared entities and temporal bucketing (week)
        for m in memories:
            entities = memory_entities.get(m['id'], [])
            created = m.get('created_at')
            
            # Extract year-week bucket (e.g. 2024-W15)
            week_bucket = "unknown"
            if created:
                try:
                    native_dt = getattr(created, 'to_native', lambda: created)()
                    if hasattr(native_dt, 'isocalendar'):
                        iso = native_dt.isocalendar()
                        week_bucket = f"{iso[0]}-W{iso[1]:02d}"
                except Exception:
                    pass

            for entity in entities[:2]:  # Use top 2 entities as key
                if entity:
                    key = f"topic:{entity}|week:{week_bucket}"
                    if key not in clusters:
                        clusters[key] = []
                    # Compare by ID to avoid dict identity issues
                    if m['id'] not in {x['id'] for x in clusters[key]}:
                        clusters[key].append(m)

        # Merge container clusters if they're small
        for container, cluster_memories in container_clusters.items():
            if len(cluster_memories) >= 2:
                clusters[f"container:{container}"] = cluster_memories

        return clusters

    def _extract_facts_from_cluster(self, memories: List[Dict]) -> List[Dict]:
        """Extract structured facts from a cluster of memories"""
        facts = []

        # Simple regex-based extraction
        import re

        for m in memories:
            content = m.get('content', '')

            # Extract dates
            dates = re.findall(r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?\b', content, re.I)
            for date in dates[:2]:
                facts.append({
                    'subject': 'conversation',
                    'predicate': 'mentioned_date',
                    'value': date.strip(),
                    'confidence': 0.7,
                    'source': m['id']
                })

            # Extract "want to" / "need to" patterns
            wants = re.findall(r'(?:want|need|should|must|going to)\s+(?:to\s+)?(.+?)[.!?]', content, re.I)
            for want in wants[:2]:
                facts.append({
                    'subject': 'user',
                    'predicate': 'expressed_intention',
                    'value': want.strip(),
                    'confidence': 0.6,
                    'source': m['id']
                })

            # Extract stated preferences
            likes = re.findall(r'(?:like|love|enjoy|prefer)s?\s+(?:to\s+)?(.+?)[.!?]', content, re.I)
            for like in likes[:2]:
                facts.append({
                    'subject': 'user',
                    'predicate': 'likes',
                    'value': like.strip(),
                    'confidence': 0.8,
                    'source': m['id']
                })

        return facts[:10]  # Limit facts per cluster

    def _store_consolidated_memory(self, summary: str, 
                                   originals: List[Dict],
                                   topic: str) -> str:
        """Store a consolidated summary memory"""
        memory_id = str(uuid.uuid4())
        original_ids = [m['id'] for m in originals]

        # Use first memory as template
        first = originals[0]

        # Calculate consolidation count
        max_prev_count = 0
        for m in originals:
            meta = m.get('metadata', {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            if meta and isinstance(meta, dict):
                max_prev_count = max(max_prev_count, meta.get('consolidation_count', 0))
        
        consolidation_count = max_prev_count + 1

        # Generate embedding
        embedding = None
        try:
            embedding = self.brain._embed(summary)
        except Exception as e:
            logger.warning(f"Failed to generate embedding for summary: {e}")

        query = """
        CREATE (m:Memory {
            id: $id,
            content: $content,
            summary: $summary,
            content_type: "consolidated",
            container: $container,
            importance: 0.6,
            salience_tags: ["consolidated", "summary"],
            timestamp: datetime(),
            source: "auto_consolidation",
            metadata: $metadata,
            archived: false,
            created_at: datetime()
        })
        WITH m
        CALL {
            WITH m
            MATCH (orig:Memory)-[:MENTIONS]->(e:Entity)
            WHERE orig.id IN $original_ids
            WITH m, e, count(orig) as mention_count
            MERGE (m)-[r:MENTIONS]->(e)
            ON CREATE SET r.weight = coalesce(r.weight, 0.3) + (0.1 * mention_count), r.confidence = 0.8
            ON MATCH SET r.weight = coalesce(r.weight, 0.3) + (0.1 * mention_count)
            RETURN count(e) as copied
        }
        WITH m
        CALL {
            WITH m 
            WITH m WHERE $embedding IS NOT NULL AND size($embedding) > 0
            SET m.embedding = $embedding
            RETURN count(m) as emb_set
        }
        RETURN m.id as id
        """

        self.brain.conn.execute_write(query, {
            'id': memory_id,
            'content': summary,
            'summary': summary[:200],
            'container': first.get('container', 'default'),
            'original_ids': original_ids,
            'embedding': embedding if embedding else [],
            'metadata': json.dumps({
                'original_count': len(originals),
                'original_ids': original_ids,
                'topic': topic,
                'consolidation_count': consolidation_count
            })
        })

        # Link to original entities (but don't duplicate)
        # Original memories still have their entity links

        logger.info(f"Stored consolidated memory {memory_id}")
        return memory_id

    def _store_fact(self, fact: Dict):
        """Store an extracted fact"""
        fact_id = str(uuid.uuid4())

        query = """
        MERGE (s:Entity {name: $subject})
        WITH s
        CREATE (f:Fact {
            id: $id,
            subject: $subject,
            predicate: $predicate,
            value: $value,
            confidence: $confidence,
            source_memory: $source,
            is_current: true,
            created_at: datetime()
        })
        CREATE (f)-[:ABOUT]->(s)
        """

        self.brain.conn.execute_write(query, {
            'id': fact_id,
            'subject': fact.get('subject', 'unknown'),
            'predicate': fact.get('predicate', 'relates_to'),
            'value': fact.get('value', ''),
            'confidence': fact.get('confidence', 0.5),
            'source': fact.get('source', '')
        })

    def _archive_memory(self, memory_id: str):
        """Archive a memory, keeping entity links"""
        query = """
        MATCH (m:Memory {id: $id})
        SET m.archived = true,
            m.archive_date = date(),
            m.embedding = null
        """

        self.brain.conn.execute_write(query, {'id': memory_id})


class RelationshipDecay:
    """
    Manages relationship weight decay to keep the graph clean.
    Weak connections are periodically decayed and removed.
    """

    def __init__(self, brain: Neo4jBrain):
        self.brain = brain

    def decay_weak_connections(self, 
                               decay_factor: float = 0.95,
                               min_threshold: float = 0.1,
                               dry_run: bool = False) -> Dict:
        """
        Decay all relationship weights and prune very weak ones.

        Args:
            decay_factor: Multiply all weights by this
            min_threshold: Remove relationships below this
            dry_run: If True, only report

        Returns:
            Statistics dict
        """
        stats = {
            'decayed': 0,
            'pruned': 0,
            'total_before': 0
        }

        # Count before
        count_query = """
        MATCH ()-[r]->()
        WHERE r.weight IS NOT NULL
        RETURN count(r) as total
        """
        result = self.brain.conn.execute_query(count_query)
        stats['total_before'] = result[0]['total'] if result else 0

        if dry_run:
            # Just report what would happen
            query = """
            MATCH ()-[r]->()
            WHERE r.weight IS NOT NULL
            RETURN count(r) as decayed, 
                   sum(CASE WHEN r.weight * $factor < $threshold THEN 1 ELSE 0 END) as pruned
            """
            result = self.brain.conn.execute_query(query, {
                'factor': decay_factor, 
                'threshold': min_threshold
            })
            if result:
                stats['decayed'] = result[0].get('decayed', 0)
                stats['pruned'] = result[0].get('pruned', 0)
            return stats

        # Decay all weights
        decay_query = """
        MATCH ()-[r]->()
        WHERE r.weight IS NOT NULL
        SET r.weight = r.weight * $factor
        RETURN count(r) as decayed
        """
        result = self.brain.conn.execute_write(decay_query, {'factor': decay_factor})
        if result:
            stats['decayed'] = result[0].get('decayed', 0) if hasattr(result[0], 'get') else len(result)

        # Count weak relationships before pruning
        count_weak_query = """
        MATCH ()-[r]->()
        WHERE r.weight IS NOT NULL AND r.weight < $threshold
        RETURN count(r) as pruned
        """
        result = self.brain.conn.execute_query(count_weak_query, {'threshold': min_threshold})
        if result:
            stats['pruned'] = result[0].get('pruned', 0) if hasattr(result[0], 'get') else 0

        # Prune very weak relationships
        prune_query = """
        MATCH ()-[r]->()
        WHERE r.weight IS NOT NULL AND r.weight < $threshold
        DELETE r
        """
        self.brain.conn.execute_write(prune_query, {'threshold': min_threshold})

        logger.info(f"Relationship decay complete: {stats}")
        return stats

    def decay_by_age(self, older_than_days: int = 30,
                     decay_factor: float = 0.8) -> Dict:
        """
        Extra decay for old relationships (use them or lose them).
        """
        query = """
        MATCH ()-[r]->()
        WHERE r.updated_at IS NOT NULL
          AND r.updated_at < datetime() - duration({days: $days})
        SET r.weight = r.weight * $factor
        RETURN count(r) as decayed
        """

        result = self.brain.conn.execute_write(query, {
            'days': older_than_days,
            'factor': decay_factor
        })

        return {
            'old_connections_decayed': result[0].get('decayed', 0) if result else 0
        }


def create_consolidator(brain: Neo4jBrain = None) -> MemoryConsolidator:
    """Factory function"""
    if brain is None:
        brain = Neo4jBrain()
    return MemoryConsolidator(brain)


def create_decay(brain: Neo4jBrain = None) -> RelationshipDecay:
    """Factory function"""
    if brain is None:
        brain = Neo4jBrain()
    return RelationshipDecay(brain)

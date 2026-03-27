
"""
Agent Zero Brain Integration

Usage:
    from memory.brain_integration import brain_agent, observe, respond, get_context, recall

    # On user message:
    observe("user message here")

    # Before generating response:
    context = get_context()

    # After response:
    respond("assistant response here")
"""

import os
import uuid
from datetime import datetime
from typing import Optional, Dict, List

from .brain import Neo4jBrain
from .connection import Neo4jConnection


class BrainAgent:
    """Simple brain agent for Agent Zero integration."""

    _instance = None

    def __init__(self):
        self.brain: Optional[Neo4jBrain] = None
        self._initialized = False
        self._conversation_context: List[Dict] = []
        self._grounded_context = ""

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self) -> bool:
        if self._initialized:
            return True
        try:
            from .connection import create_driver

            driver = create_driver()
            driver.verify_connectivity()

            # Setup central Neo4jBrain instance with shared connection
            conn = Neo4jConnection(driver)
            self.brain = Neo4jBrain(connection=conn)

            self._initialized = True
            return True
        except Exception as e:
            print(f"Brain init failed: {e}")
            return False

    def observe(self, message: str, role: str = "user") -> Optional[str]:
        if not self._initialized:
            if not self.initialize():
                return None

        self._conversation_context.append({'role': role, 'content': message})

        # Store memory through the central Neo4jBrain (with embeddings + entities)
        memory_id = None
        if self.brain:
            try:
                memory_id = self.brain.add_memory(
                    content=message,
                    content_type='conversation',
                    importance=0.5,
                    container='default',
                    source='agent_zero'
                )
            except Exception as e:
                print(f"Store memory failed: {e}")

        self._update_grounded_context()
        return memory_id

    def respond(self, response: str) -> Optional[str]:
        if not self._initialized or not self.brain:
            return None
        self._conversation_context.append({'role': 'assistant', 'content': response})

        try:
            return self.brain.add_memory(
                content=response,
                content_type='conversation',
                importance=0.3,
                container='default',
                source='agent_zero_response'
            )
        except Exception as e:
            print(f"Store response failed: {e}")
            return None

    def _update_grounded_context(self):
        if not self._conversation_context or not self.brain:
            self._grounded_context = ""
            return

        # Extract entities from recent messages using Neo4jBrain
        recent_entities: List[str] = []
        for msg in self._conversation_context[-3:]:
            entities = self.brain.extract_entities(msg['content'])
            recent_entities.extend([e.get('name', '') for e in entities])
        recent_entities = [e for e in list(set(recent_entities)) if e][:5]

        if not recent_entities:
            self._grounded_context = ""
            return

        # Find related memories via Neo4jBrain graph queries
        proactive: List[Dict] = []
        for entity in recent_entities:
            try:
                memories = self.brain.get_related_memories(entity, top_k=3)
                for mem in memories:
                    if mem.get('summary'):
                        proactive.append({
                            'entity': entity,
                            'memory': mem['summary'],
                            'strength': mem.get('importance', 0.5)
                        })
            except Exception as e:
                print(f"Grounded context lookup failed for {entity}: {e}")

        if proactive:
            proactive.sort(key=lambda x: x['strength'], reverse=True)
            seen: set = set()
            unique = [p for p in proactive if p['memory'] not in seen and not seen.add(p['memory'])]

            parts = ["**Grounded Context:**"]
            for p in unique[:5]:
                parts.append(f"• {p['memory']} (related to {p['entity']})")

            self._grounded_context = "\n".join(parts)
        else:
            self._grounded_context = ""

    def get_context(self) -> str:
        return self._grounded_context

    def recall(self, topic: str, limit: int = 5) -> List[Dict]:
        if not self.brain:
            return []

        results: List[Dict] = []
        try:
            entities = self.brain.extract_entities(topic)
            for entity in entities:
                mems = self.brain.get_related_memories(entity.get('name', ''), top_k=limit)
                for mem in mems:
                    results.append({
                        'topic': entity.get('name', ''),
                        'memory': mem.get('summary', ''),
                        'importance': mem.get('importance', 0.5)
                    })
        except Exception as e:
            print(f"Recall failed: {e}")

        # Remove duplicates
        unique: List[Dict] = []
        seen: set = set()
        for r in results:
            if r['memory'] not in seen:
                seen.add(r['memory'])
                unique.append(r)

        return unique[:limit]

    def search(self, query: str) -> List[Dict]:
        if not self.brain:
            return []

        try:
            semantic_results = self.brain.semantic_search(query, top_k=10)
            return [
                {
                    'id': r.get('id'),
                    'summary': r.get('summary'),
                    'importance': r.get('importance', 0.5)
                }
                for r in semantic_results
            ]
        except Exception as e:
            print(f"Search failed: {e}")
            return []

    def get_stats(self) -> Dict:
        """Delegate to Neo4jBrain.get_stats() — single source of truth."""
        if not self._initialized or not self.brain:
            return {'status': 'not_initialized'}

        try:
            stats = self.brain.get_stats()
            stats['initialized'] = True
            return stats
        except Exception as e:
            return {'initialized': True, 'error': str(e)}

    def start_conversation(self):
        self._conversation_context = []
        self._grounded_context = ""


# Singleton
brain_agent = BrainAgent.get_instance()


# Convenience functions
def observe(message: str, role: str = "user") -> Optional[str]:
    return brain_agent.observe(message, role)

def respond(response: str) -> Optional[str]:
    return brain_agent.respond(response)

def get_context() -> str:
    return brain_agent.get_context()

def recall(topic: str, limit: int = 5) -> List[Dict]:
    return brain_agent.recall(topic, limit)

def search(query: str) -> List[Dict]:
    return brain_agent.search(query)

def get_stats() -> Dict:
    return brain_agent.get_stats()


# =====================================================
# MEMORY CONSOLIDATION FUNCTIONS
# =====================================================

def consolidate_memories(older_than_days: int = 7, dry_run: bool = False) -> Dict:
    """
    Consolidate old memories into summaries.
    Delegates to the MemoryConsolidator from consolidation.py.
    """
    if not brain_agent._initialized or not brain_agent.brain:
        return {'error': 'Brain not initialized'}

    try:
        from .consolidation import MemoryConsolidator
        consolidator = MemoryConsolidator(brain_agent.brain)
        result = consolidator.consolidate_old_memories(
            older_than_days=older_than_days,
            dry_run=dry_run
        )
        return {
            'memories_processed': result.memories_processed,
            'memories_archived': result.memories_archived,
            'memories_consolidated': result.memories_consolidated,
            'facts_extracted': result.facts_extracted,
            'summaries': result.new_summaries
        }
    except Exception as e:
        return {'error': str(e)}


def decay_connections(threshold: float = 0.1, dry_run: bool = False) -> Dict:
    """
    Decay weak relationship weights and prune very weak ones.
    Delegates to the RelationshipDecay from consolidation.py.
    """
    if not brain_agent._initialized or not brain_agent.brain:
        return {'error': 'Brain not initialized'}

    try:
        from .consolidation import RelationshipDecay
        decay = RelationshipDecay(brain_agent.brain)
        return decay.decay_weak_connections(
            min_threshold=threshold,
            dry_run=dry_run
        )
    except Exception as e:
        return {'error': str(e)}


def get_maintenance_status() -> Dict:
    """
    Get status of memories that need consolidation.
    Uses Neo4jBrain connection instead of raw sessions.
    """
    if not brain_agent._initialized or not brain_agent.brain:
        return {'error': 'Brain not initialized'}

    brain = brain_agent.brain

    try:
        # Old memories count
        old_result = brain.conn.execute_single("""
            MATCH (m:Memory)
            WHERE m.archived = false
              AND m.created_at < datetime() - duration({days: 7})
              AND m.content_type IN ["conversation", "note"]
            RETURN count(m) as c
        """)

        # Archived count
        archived_result = brain.conn.execute_single("""
            MATCH (m:Memory) WHERE m.archived = true RETURN count(m) as c
        """)

        # Weak relationships
        weak_result = brain.conn.execute_single("""
            MATCH ()-[r]->() WHERE r.weight IS NOT NULL AND r.weight < 0.2 RETURN count(r) as c
        """)

        old_count = old_result.get('c', 0) if old_result else 0
        archived_count = archived_result.get('c', 0) if archived_result else 0
        weak_count = weak_result.get('c', 0) if weak_result else 0

        return {
            'old_memories': old_count,
            'archived_memories': archived_count,
            'weak_connections': weak_count,
            'consolidation_needed': old_count > 5
        }
    except Exception as e:
        return {'error': str(e)}

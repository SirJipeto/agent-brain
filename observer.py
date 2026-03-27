"""
Observer Framework: The proactive brain loop

This implements the "Inner Thoughts" pattern where every user message
triggers a background process that:
1. Extracts entities
2. Traverses the knowledge graph
3. Evaluates relevance
4. Queues memories for potential surfacing

This enables natural memory references like:
"Since you mentioned X, don't forget that Y..."
"This reminds me of Z from our conversation about..."
"""

import uuid
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import logging

from .brain import Neo4jBrain

logger = logging.getLogger(__name__)


@dataclass
class ConversationalEvent:
    """A single conversational event (user message)"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    entities: List[Dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    context_tags: List[str] = field(default_factory=list)
    processed: bool = False


@dataclass
class ProactiveMemory:
    """A memory surfaced proactively for potential injection"""
    memory_id: str
    content: str
    connection_explanation: str  # How it relates to current conversation
    connection_strength: float
    suggested_mention: str  # Natural language way to bring it up
    source_entity: str
    importance: float
    timestamp: datetime


class RelevanceEvaluator:
    """
    Evaluates whether a retrieved memory is relevant enough
    to surface proactively.
    """

    def __init__(self, brain: Neo4jBrain):
        self.brain = brain
        self.recently_surfaced = deque(maxlen=20)  # Memory IDs surfaced recently
        self.topic_cooldown = {}  # Entity -> last mentioned time

    def is_relevant_enough(self, candidate: Dict, 
                          current_event: ConversationalEvent,
                          current_entities: List[str]) -> bool:
        """
        Multi-factor relevance evaluation.
        """
        # Factor 1: Connection strength threshold
        strength = candidate.get('connection_strength', 0.5)
        if strength < 0.25:
            return False

        # Factor 2: Memory importance
        importance = candidate.get('memory', {}).get('importance', 0.5)
        if importance > 0.8:
            return True  # High importance always surfaces

        # Factor 3: Was recently mentioned?
        memory_id = candidate.get('memory', {}).get('id')
        if memory_id in self.recently_surfaced:
            return False

        # Factor 4: Connection is through current topic
        source_entity = candidate.get('source_entity', '')
        if source_entity in current_entities:
            return True

        # Factor 5: Salience tags match current context
        memory_tags = candidate.get('memory', {}).get('salience_tags', [])
        if any(tag in current_event.context_tags for tag in memory_tags):
            return True

        # Factor 6: User goal alignment (stored in profile)
        user_goals = self.brain.get_user_goals()
        if user_goals:
            memory_content = candidate.get('memory', {}).get('content', '').lower()
            if any(goal.lower() in memory_content for goal in user_goals):
                return True

        # Default: moderate threshold
        return strength >= 0.4 and importance >= 0.5

    def mark_surfaced(self, memory_id: str):
        """Track surfaced memories to avoid repetition"""
        self.recently_surfaced.append(memory_id)

    def get_cooldown(self, entity: str, minutes: int = 5) -> bool:
        """Check if entity is in cooldown period"""
        last = self.topic_cooldown.get(entity)
        if last and (datetime.now() - last) < timedelta(minutes=minutes):
            return True
        return False

    def set_cooldown(self, entity: str):
        """Mark entity as recently mentioned"""
        self.topic_cooldown[entity] = datetime.now()


class ContextBuilder:
    """
    Builds natural language context for injection into LLM prompts.
    """

    @staticmethod
    def build_surface_prompt(memory: ProactiveMemory) -> str:
        """
        Convert a proactive memory into a natural prompt.
        """
        if memory.suggested_mention:
            return memory.suggested_mention

        # Build natural reference
        if memory.connection_explanation:
            return f"{memory.source_entity}: {memory.connection_explanation}"

        return f"Related: {memory.content[:200]}..."

    @staticmethod
    def build_grounded_context(proactive_memories: List[ProactiveMemory],
                               implicit_connections: List[Dict] = None) -> str:
        """
        Build full grounded context string for system prompt.
        """
        if not proactive_memories:
            return ""

        parts = ["**Grounded Context:**"]

        for mem in proactive_memories[:5]:  # Limit to 5 for prompt size
            surface = ContextBuilder.build_surface_prompt(mem)
            parts.append(f"• {surface}")

        if implicit_connections:
            parts.append("\n**Implicit Connections:**")
            for conn in implicit_connections[:3]:
                path = " → ".join(conn.get('path', []))
                parts.append(f"• {conn.get('source')} relates to {conn.get('target')} through: {path}")

        return "\n".join(parts)


class ObserverFramework:
    """
    Main observer class implementing the proactive brain loop.

    Usage:
        brain = Neo4jBrain()
        observer = ObserverFramework(brain)

        # On every user message:
        observer.on_message("I'm planning a trip to Japan")

        # Before generating response:
        context = observer.get_grounded_context()
    """

    def __init__(self, brain: Neo4jBrain):
        self.brain = brain
        self.evaluator = RelevanceEvaluator(brain)
        self.conversation_context = deque(maxlen=10)
        self.recent_entities = []
        self._proactive_queue = []
        self._current_event = None
        self._hooks = []  # Callbacks for memory events

    def on_message(self, message: str, 
                  container: str = "default",
                  source: str = "user_message",
                  extract_entities: bool = True,
                  store_memory: bool = True) -> Optional[str]:
        """
        Process a new conversational event.

        This is the main entry point - call this on every user message.

        Args:
            message: The user's message
            container: Context grouping
            source: Origin identifier
            extract_entities: Whether to extract entities
            store_memory: Whether to store as memory

        Returns:
            Memory ID if stored, None otherwise
        """
        # 1. Create event
        event = ConversationalEvent(
            content=message,
            timestamp=datetime.now()
        )

        # 2. Extract entities
        if extract_entities:
            event.entities = self.brain.extract_entities(message)
            self.recent_entities = [e['name'] for e in event.entities]

        # 3. Update conversation context
        self.conversation_context.append(event)
        self._current_event = event

        # 4. Trigger proactive memory retrieval
        proactive = self.get_proactive_memories(event)
        self._proactive_queue.extend(proactive)

        # 5. Store as memory
        memory_id = None
        if store_memory:
            memory_id = self.brain.add_memory(
                content=message,
                content_type="conversation",
                container=container,
                source=source,
                entities=event.entities
            )

        # 6. Trigger hooks
        self._trigger_hooks('on_message', event)

        return memory_id

    def get_proactive_memories(self, event: ConversationalEvent) -> List[ProactiveMemory]:
        """
        Retrieve memories that might be relevant to the current conversation.
        This implements the core proactive recall.
        """
        candidates = []
        current_entity_names = [e['name'] for e in event.entities]

        for entity_name in current_entity_names:
            # Skip if in cooldown
            if self.evaluator.get_cooldown(entity_name):
                continue

            # Get connected entities through graph traversal
            connections = self.brain.graph_traverse(entity_name, depth=3)

            for conn in connections:
                # Find memories mentioning the connected entity
                related_memories = self.brain.get_related_memories(
                    conn.get('connected_name', ''), 
                    top_k=3
                )

                for memory in related_memories:
                    # Get implicit connection path
                    path = self.brain.find_implicit_connections(
                        entity_name, 
                        conn.get('connected_name', '')
                    )

                    candidate = {
                        'memory': memory,
                        'connection_strength': conn.get('strength', 0.5),
                        'source_entity': entity_name,
                        'connection_path': path[0] if path else None,
                        'connection_explanation': self._explain_connection(
                            entity_name,
                            conn.get('connected_name', ''),
                            conn.get('relationship_types', [])
                        )
                    }

                    if self.evaluator.is_relevant_enough(candidate, event, 
                                                          current_entity_names):
                        candidates.append(ProactiveMemory(
                            memory_id=memory.get('id', ''),
                            content=memory.get('content', ''),
                            connection_explanation=candidate['connection_explanation'],
                            connection_strength=candidate['connection_strength'],
                            suggested_mention=self._build_suggested_mention(
                                memory, entity_name, conn.get('connected_name', '')
                            ),
                            source_entity=entity_name,
                            importance=memory.get('importance', 0.5),
                            timestamp=memory.get('timestamp', datetime.now())
                        ))

        # Sort by strength and importance
        candidates.sort(key=lambda x: (x.connection_strength * x.importance), reverse=True)

        # Mark as surfaced
        for candidate in candidates[:5]:
            self.evaluator.mark_surfaced(candidate.memory_id)
            self.evaluator.set_cooldown(candidate.source_entity)

        return candidates[:5]  # Return top 5

    def _explain_connection(self, entity_a: str, entity_b: str, 
                           rel_types: List[str]) -> str:
        """Generate natural explanation of connection"""
        if rel_types:
            rel = rel_types[0] if rel_types else 'related'
            return f"{entity_a} is {rel.lower()} to {entity_b}"
        return f"{entity_a} relates to {entity_b}"

    def _build_suggested_mention(self, memory: Dict, 
                                current_entity: str,
                                connected_entity: str) -> str:
        """Build a natural way to mention this memory"""
        content = memory.get('content', '')[:100]

        templates = [
            f"Since you mentioned {current_entity}, you might remember: {content}...",
            f"This connects to what we discussed about {connected_entity}: {content}...",
            f"Speaking of {current_entity}, I recall: {content}..."
        ]

        import random
        return random.choice(templates)

    def get_grounded_context(self, max_memories: int = 5) -> str:
        """
        Get context string for injection into LLM prompt.
        Call this before generating a response.
        """
        if not self._proactive_queue:
            return ""

        memories_to_include = self._proactive_queue[:max_memories]

        # Get implicit connections for additional context
        if self.recent_entities:
            implicit = []
            for i, e1 in enumerate(self.recent_entities[:3]):
                for e2 in self.recent_entities[i+1:]:
                    conns = self.brain.find_implicit_connections(e1, e2)
                    if conns:
                        implicit.append({
                            'source': e1,
                            'target': e2,
                            'path': conns[0].get('path', [])
                        })
        else:
            implicit = None

        return ContextBuilder.build_grounded_context(memories_to_include, implicit)

    def clear_queue(self):
        """Clear the proactive memory queue"""
        self._proactive_queue = []

    def _trigger_hooks(self, event_type: str, data: Any):
        """Trigger registered hooks"""
        for hook in self._hooks:
            try:
                if callable(hook):
                    hook(event_type, data)
            except Exception as e:
                logger.warning(f"Hook error: {e}")

    def register_hook(self, callback: Callable):
        """Register a callback for memory events"""
        self._hooks.append(callback)

    # ============================================
    # CONVERSATION MANAGEMENT
    # ============================================

    def start_conversation(self, topic: str = None, container: str = "default"):
        """Start a new conversation context"""
        self.conversation_context.clear()
        self._proactive_queue.clear()
        self.recent_entities = []

        if topic:
            self.on_message(topic, container=container, 
                          source="conversation_start", store_memory=False)

    def end_conversation(self):
        """End current conversation - clear context"""
        # Optionally consolidate recent memories
        recent_messages = [e.content for e in self.conversation_context]

        if len(recent_messages) > 2:
            # Generate conversation summary
            summary = self.brain.add_memory(
                content=" | ".join(recent_messages),
                content_type="conversation_summary",
                source="auto_consolidation"
            )

        self.conversation_context.clear()
        self._proactive_queue.clear()
        self.recent_entities = []

    def get_conversation_summary(self) -> Dict:
        """Get summary of current conversation"""
        return {
            'message_count': len(self.conversation_context),
            'entities_mentioned': self.recent_entities,
            'proactive_surfaces': len(self._proactive_queue),
            'recent_messages': [e.content for e in list(self.conversation_context)[-5:]]
        }

    # ============================================
    # SPONTANEOUS RECALL (Manual Trigger)
    # ============================================

    def recall_related(self, topic: str, depth: int = 2) -> Dict:
        """
        Manual trigger for associative recall on a topic.
        Returns rich context including associations and connections.
        """
        # Semantic search
        semantic = self.brain.semantic_search(topic, top_k=5)

        # Graph associations
        associations = self.brain.recall_associations(topic, depth=depth)

        # Activation spreading
        entities = self.brain.extract_entities(topic)
        entity_names = [e['name'] for e in entities[:5]]
        activated = self.brain.spread_activation(entity_names)

        return {
            'topic': topic,
            'semantic_matches': semantic,
            'associations': associations[:10],
            'activated_concepts': list(activated.items())[:10],
            'entity_extractions': entities
        }


def create_observer(brain: Neo4jBrain = None) -> ObserverFramework:
    """Convenience factory"""
    if brain is None:
        from .brain import Neo4jBrain
        brain = Neo4jBrain()
    return ObserverFramework(brain)

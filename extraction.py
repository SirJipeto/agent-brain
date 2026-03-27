"""
LLM-based Entity and Fact Extractor

Replaces simple regex extraction with intelligent LLM-based parsing.
Uses structured output for reliable entity extraction.
"""

import json
import re
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """A single extracted entity"""
    name: str
    entity_type: str  # person, place, organization, concept, technology, event, etc.
    description: str = ""
    confidence: float = 0.8
    properties: Optional[Dict] = None


@dataclass
class ExtractedFact:
    """A structured fact extracted from text"""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.7
    context: str = ""


@dataclass
class ExtractionResult:
    """Complete extraction result"""
    entities: List[ExtractedEntity]
    facts: List[ExtractedFact]
    summary: str = ""
    topics: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)  # user's goals, questions, statements


class EntityExtractor:
    """
    LLM-powered entity and fact extraction.

    Uses structured prompting to extract:
    - Named entities (people, places, organizations)
    - Concepts and topics
    - Relationships between entities
    - User intents and goals
    """

    # Entity type mapping
    ENTITY_TYPES = {
        'person': 'Person', 
        'place': 'Location',
        'organization': 'Organization',
        'technology': 'Technology',
        'concept': 'Concept',
        'event': 'Event',
        'date': 'Temporal',
        'object': 'Object',
        'document': 'Document',
        'project': 'Project'
    }

    def __init__(self, llm_callable: Optional[Callable] = None):
        """
        Args:
            llm_callable: Function(content: str, schema: dict) -> dict
                        If None, uses mock extraction for testing
        """
        self._llm = llm_callable

    def extract(self, text: str, include_facts: bool = True) -> ExtractionResult:
        """
        Main extraction method.

        Args:
            text: Text to extract from
            include_facts: Whether to extract facts (slower)

        Returns:
            ExtractionResult with entities, facts, summary
        """
        if not text or len(text.strip()) < 3:
            return ExtractionResult(entities=[], facts=[], summary="")

        if self._llm:
            return self._extract_with_llm(text, include_facts)
        else:
            return self._simple_extract(text)

    def _extract_with_llm(self, text: str, include_facts: bool) -> ExtractionResult:
        """Extract using LLM with structured output"""

        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": list(self.ENTITY_TYPES.keys())},
                            "description": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["name", "type"]
                    }
                },
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["subject", "predicate", "object"]
                    }
                },
                "summary": {"type": "string"},
                "topics": {"type": "array", "items": {"type": "string"}},
                "intents": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["entities", "summary"]
        }

        prompt = f"""Extract structured information from this text:

TEXT:
{text[:2000]}

Extract:
1. Named entities (people, places, organizations, technologies, concepts)
2. Facts in subject-predicate-object format
3. A brief summary
4. Main topics
5. User intents/goals (what the user wants or is expressing)

Return valid JSON matching this schema:
{json.dumps(schema, indent=2)}"""

        try:
            result = self._llm(prompt, schema)

            # Convert to dataclasses
            entities = [
                ExtractedEntity(
                    name=e.get('name', ''),
                    entity_type=e.get('type', 'concept'),
                    description=e.get('description', ''),
                    confidence=e.get('confidence', 0.8)
                )
                for e in result.get('entities', [])
                if e.get('name')
            ]

            facts = [
                ExtractedFact(
                    subject=f.get('subject', ''),
                    predicate=f.get('predicate', ''),
                    object=f.get('object', ''),
                    confidence=f.get('confidence', 0.7)
                )
                for f in result.get('facts', [])
                if f.get('subject') and f.get('object')
            ]

            return ExtractionResult(
                entities=entities,
                facts=facts,
                summary=result.get('summary', ''),
                topics=result.get('topics', []),
                intents=result.get('intents', [])
            )

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}, falling back to simple")
            return self._simple_extract(text)

    def _simple_extract(self, text: str) -> ExtractionResult:
        """
        Simple rule-based extraction without LLM.
        Used for testing or when LLM is unavailable.
        """
        entities = []
        facts = []

        # Extract capitalized phrases
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*\b', text)
        seen = set()
        for word in capitalized:
            if word.lower() not in seen and len(word) > 2:
                seen.add(word.lower())
                entities.append(ExtractedEntity(
                    name=word,
                    entity_type='concept',
                    confidence=0.5
                ))

        # Extract quoted strings
        quotes = re.findall(r'"([^"]+)"', text)
        for quote in quotes[:3]:
            if quote.lower() not in seen and len(quote) > 3:
                seen.add(quote.lower())
                entities.append(ExtractedEntity(
                    name=quote,
                    entity_type='concept',
                    description='Quoted phrase',
                    confidence=0.4
                ))

        # Extract intentions
        intents = []
        want_patterns = [r'want to (.+)', r'need to (.+)', r'going to (.+)']
        for pattern in want_patterns:
            matches = re.findall(pattern, text, re.I)
            intents.extend([f"plans to {m}" for m in matches[:2]])

        # Simple summary
        words = text.split()[:30]
        summary = ' '.join(words) + ('...' if len(text.split()) > 30 else '')

        # Topics (most common words excluding stopwords)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for', 'on', 'and', 'or', 'but', 'i', 'you', 'we', 'they', 'my', 'your', 'this', 'that', 'it'}
        word_freq = {}
        for word in text.lower().split():
            if word not in stopwords and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        topics = sorted(word_freq.keys(), key=word_freq.get, reverse=True)[:5]

        return ExtractionResult(
            entities=entities[:10],
            facts=facts,
            summary=summary,
            topics=topics,
            intents=intents
        )

    def extract_entities_only(self, text: str) -> List[ExtractedEntity]:
        """Fast entity extraction without facts"""
        result = self.extract(text, include_facts=False)
        return result.entities

    def extract_facts_only(self, text: str) -> List[ExtractedFact]:
        """Extract only facts"""
        result = self.extract(text, include_facts=True)
        return result.facts


class ConversationAnalyzer:
    """
    Analyzes conversational context to extract:
    - User goals and intentions
    - Emotional tone
    - Action items
    - Follow-up topics
    """

    def __init__(self, llm_callable: Optional[Callable] = None):
        self.extractor = EntityExtractor(llm_callable)
        self._llm = llm_callable

    def analyze_conversation(self, messages: List[Dict]) -> Dict:
        """
        Analyze a conversation (list of message dicts with 'role' and 'content').

        Returns:
            Dict with goals, action_items, follow_ups, tone
        """
        if not messages:
            return {
                'goals': [],
                'action_items': [],
                'follow_ups': [],
                'tone': 'neutral',
                'entities': [],
                'summary': ''
            }

        # Combine messages
        combined = "\n".join([
            f"{m.get('role', 'user')}: {m.get('content', '')}" 
            for m in messages[-10:]  # Last 10 messages
        ])

        if self._llm:
            return self._analyze_with_llm(combined)
        else:
            return self._simple_analyze(combined)

    def _analyze_with_llm(self, text: str) -> Dict:
        """LLM-powered analysis"""

        prompt = f"""Analyze this conversation:

{text[:1500]}

Extract:
1. User goals (what the user is trying to accomplish)
2. Action items (tasks or follow-ups mentioned)
3. Follow-up topics (what might be worth discussing next)
4. Emotional tone (positive, negative, neutral, frustrated, excited)
5. Key entities mentioned
6. Brief summary

Return JSON:"""

        schema = {
            "type": "object",
            "properties": {
                "goals": {"type": "array", "items": {"type": "string"}},
                "action_items": {"type": "array", "items": {"type": "string"}},
                "follow_ups": {"type": "array", "items": {"type": "string"}},
                "tone": {"type": "string"},
                "entities": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"}
            }
        }

        try:
            result = self._llm(prompt, schema)
            return {
                'goals': result.get('goals', []),
                'action_items': result.get('action_items', []),
                'follow_ups': result.get('follow_ups', []),
                'tone': result.get('tone', 'neutral'),
                'entities': result.get('entities', []),
                'summary': result.get('summary', '')
            }
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return self._simple_analyze(text)

    def _simple_analyze(self, text: str) -> Dict:
        """Simple keyword-based analysis"""
        goals = []
        action_items = []

        # Extract goals
        goal_patterns = [
            (r'i want to (.+)', 'wants to {}'),
            (r'i need to (.+)', 'needs to {}'),
            (r'i\'m trying to (.+)', 'trying to {}'),
            (r'my goal is (.+)', 'goal: {}'),
        ]

        for pattern, template in goal_patterns:
            matches = re.findall(pattern, text, re.I)
            for m in matches[:2]:
                goals.append(template.format(m.strip()))

        # Extract action items
        action_patterns = [
            (r'(?:remind me to|remember to|don\'t forget to) (.+)', 'reminder: {}'),
            (r'i should (.+)', 'should: {}'),
            (r'(?:todo|task): (.+)', 'todo: {}'),
        ]

        for pattern, template in action_patterns:
            matches = re.findall(pattern, text, re.I)
            for m in matches[:2]:
                action_items.append(template.format(m.strip()))

        # Detect tone
        positive = sum(1 for w in ['great', 'awesome', 'love', 'perfect', 'thanks', 'happy'] if w in text.lower())
        negative = sum(1 for w in ['frustrated', 'annoyed', 'angry', 'disappointed', 'wrong', 'bad'] if w in text.lower())
        tone = 'positive' if positive > negative else ('negative' if negative > positive else 'neutral')

        # Simple summary
        summary = ' '.join(text.split()[:20]) + '...'

        return {
            'goals': goals,
            'action_items': action_items,
            'follow_ups': [],
            'tone': tone,
            'entities': [],
            'summary': summary
        }


def create_extractor(llm_callable: Optional[Callable] = None) -> EntityExtractor:
    """Factory function"""
    return EntityExtractor(llm_callable)


def create_analyzer(llm_callable: Optional[Callable] = None) -> ConversationAnalyzer:
    """Factory function"""
    return ConversationAnalyzer(llm_callable)

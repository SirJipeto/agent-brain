"""
Tests for entity extraction (brain.py:extract_entities and extraction.py).

Covers:
- spaCy NER extraction (primary)
- Regex fallback extraction
- Entity type classification
- Edge cases (empty text, special characters, long text)
- ExtractionResult dataclass behavior
"""

import pytest
from agent_brain.brain import Neo4jBrain
from agent_brain.extraction import (
    EntityExtractor,
    ConversationAnalyzer,
    ExtractedEntity,
    ExtractedFact,
    ExtractionResult,
)
from tests.conftest import FakeNeo4jConnection, mock_embedder


# ============================================================================
# brain.py:extract_entities() — spaCy NER
# ============================================================================

class TestBrainExtractEntities:
    """Tests for spaCy NER extraction in brain.py (falls back to regex)."""

    def setup_method(self):
        self.conn = FakeNeo4jConnection()
        self.brain = Neo4jBrain(connection=self.conn, embedder=mock_embedder)

    def test_extracts_named_entities(self):
        entities = self.brain.extract_entities("Alice went to Paris with Bob")
        names = [e['name'] for e in entities]
        assert "Alice" in names
        assert "Paris" in names
        assert "Bob" in names

    def test_classifies_entity_types(self):
        """spaCy should classify entities into proper types, not just 'concept'."""
        entities = self.brain.extract_entities("Alice works at Google in Tokyo")
        types = {e['name']: e['type'] for e in entities}
        unique_types = set(types.values())
        # With spaCy, we expect at least person and organization
        assert len(unique_types) >= 2, (
            f"Expected multiple entity types with spaCy, got {unique_types}. "
            f"Entities: {entities}"
        )

    def test_person_type(self):
        entities = self.brain.extract_entities("Alice talked to Bob")
        types = {e['name']: e['type'] for e in entities}
        assert types.get("Alice") == "person"
        assert types.get("Bob") == "person"

    def test_organization_type(self):
        entities = self.brain.extract_entities("She works at Google and Microsoft")
        types = {e['name']: e['type'] for e in entities}
        assert types.get("Google") == "organization"

    def test_place_type(self):
        entities = self.brain.extract_entities("He traveled to Tokyo")
        types = {e['name']: e['type'] for e in entities}
        assert types.get("Tokyo") == "place"

    def test_extracts_quoted_phrases(self):
        entities = self.brain.extract_entities('He said "hello world" to her')
        names = [e['name'] for e in entities]
        assert "hello world" in names

    def test_deduplicates_entities(self):
        entities = self.brain.extract_entities("Alice met Alice again with Alice")
        alice_count = sum(1 for e in entities if e['name'] == 'Alice')
        assert alice_count == 1, "Should deduplicate entities"

    def test_returns_empty_for_empty_text(self):
        entities = self.brain.extract_entities("")
        assert entities == []

    def test_returns_empty_for_no_entities(self):
        entities = self.brain.extract_entities("this is all lowercase with no quotes")
        assert entities == []

    def test_has_confidence(self):
        entities = self.brain.extract_entities("Alice works at Google")
        for e in entities:
            assert 'confidence' in e
            assert 0.0 < e['confidence'] <= 1.0

    def test_special_characters_in_text(self):
        """Should not crash on special characters"""
        entities = self.brain.extract_entities("C++ is great! @Alice #Python $100 (Neo4j)")
        assert isinstance(entities, list)

    def test_unicode_text(self):
        """Should handle unicode text"""
        entities = self.brain.extract_entities("Tōkyō is a city. München is nice.")
        assert isinstance(entities, list)


# ============================================================================
# extraction.py:EntityExtractor — simple (no LLM) path, now spaCy-backed
# ============================================================================

class TestEntityExtractorSimple:
    """Tests for extraction.py's NER extraction (spaCy + regex fallback, no LLM)"""

    def setup_method(self):
        self.extractor = EntityExtractor(llm_callable=None)

    def test_returns_extraction_result(self):
        result = self.extractor.extract("Alice works at Google")
        assert isinstance(result, ExtractionResult)
        assert isinstance(result.entities, list)
        assert isinstance(result.facts, list)
        assert isinstance(result.summary, str)
        assert isinstance(result.topics, list)

    def test_extracts_named_entities(self):
        result = self.extractor.extract("Alice works at Google")
        names = [e.name for e in result.entities]
        # With spaCy, should extract Alice and Google with proper types
        assert len(names) > 0
        assert any("Alice" in n for n in names) or any("Google" in n for n in names)

    def test_extracts_quoted_strings(self):
        result = self.extractor.extract('The project is called "Agent Brain"')
        names = [e.name for e in result.entities]
        assert "Agent Brain" in names

    def test_extracts_intentions(self):
        result = self.extractor.extract("I want to learn Python programming")
        assert len(result.intents) > 0, "Should extract user intentions"

    def test_generates_summary(self):
        text = "This is a long text about various topics that should be summarized properly."
        result = self.extractor.extract(text)
        assert result.summary != ""
        assert len(result.summary) <= len(text) + 10  # Summary should be compact

    def test_extracts_topics(self):
        result = self.extractor.extract(
            "Python machine learning neural networks deep learning training data"
        )
        assert len(result.topics) > 0

    def test_entities_have_confidence(self):
        result = self.extractor.extract("Google is a company")
        for entity in result.entities:
            assert hasattr(entity, 'confidence')
            assert 0.0 <= entity.confidence <= 1.0

    def test_empty_text_returns_empty(self):
        result = self.extractor.extract("")
        assert result.entities == []
        assert result.facts == []

    def test_very_short_text_returns_empty(self):
        result = self.extractor.extract("hi")
        assert result.entities == []

    def test_limits_entities(self):
        # Generate text with many capitalized words
        words = [f"Entity{i}" for i in range(50)]
        result = self.extractor.extract(" ".join(words))
        assert len(result.entities) <= 10

    def test_extract_entities_only(self):
        entities = self.extractor.extract_entities_only("Alice met Bob")
        assert isinstance(entities, list)
        assert all(isinstance(e, ExtractedEntity) for e in entities)

    def test_extract_facts_only(self):
        facts = self.extractor.extract_facts_only("Alice met Bob in Paris")
        assert isinstance(facts, list)


# ============================================================================
# extraction.py:EntityExtractor — LLM path
# ============================================================================

class TestEntityExtractorLLM:
    """Tests for extraction.py's LLM-based extraction"""

    def test_calls_llm_when_provided(self):
        llm_called = False
        def mock_llm(prompt, schema):
            nonlocal llm_called
            llm_called = True
            return {
                'entities': [
                    {'name': 'Alice', 'type': 'person', 'confidence': 0.95},
                    {'name': 'Google', 'type': 'organization', 'confidence': 0.9}
                ],
                'facts': [
                    {'subject': 'Alice', 'predicate': 'works_at', 'object': 'Google', 'confidence': 0.85}
                ],
                'summary': 'Alice works at Google',
                'topics': ['employment'],
                'intents': []
            }

        extractor = EntityExtractor(llm_callable=mock_llm)
        result = extractor.extract("Alice works at Google")

        assert llm_called
        assert len(result.entities) == 2
        assert result.entities[0].name == 'Alice'
        assert result.entities[0].entity_type == 'person'

    def test_falls_back_on_llm_error(self):
        def failing_llm(prompt, schema):
            raise RuntimeError("LLM unavailable")

        extractor = EntityExtractor(llm_callable=failing_llm)
        result = extractor.extract("Alice works at Google")
        # Should fall back to simple extraction without crashing
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) >= 0

    def test_handles_empty_llm_response(self):
        def empty_llm(prompt, schema):
            return {}

        extractor = EntityExtractor(llm_callable=empty_llm)
        result = extractor.extract("Alice works at Google")
        assert isinstance(result, ExtractionResult)
        assert result.entities == []

    def test_filters_entities_without_name(self):
        def partial_llm(prompt, schema):
            return {
                'entities': [
                    {'name': 'Alice', 'type': 'person'},
                    {'name': '', 'type': 'person'},  # Empty name
                    {'type': 'concept'},  # No name
                ],
                'summary': 'test'
            }

        extractor = EntityExtractor(llm_callable=partial_llm)
        result = extractor.extract("Alice works at Google")
        assert len(result.entities) == 1
        assert result.entities[0].name == 'Alice'


# ============================================================================
# extraction.py:ConversationAnalyzer
# ============================================================================

class TestConversationAnalyzer:
    """Tests for conversation analysis"""

    def setup_method(self):
        self.analyzer = ConversationAnalyzer(llm_callable=None)

    def test_empty_messages(self):
        result = self.analyzer.analyze_conversation([])
        assert result['goals'] == []
        assert result['action_items'] == []
        assert result['tone'] == 'neutral'

    def test_detects_goals(self):
        messages = [
            {'role': 'user', 'content': 'I want to learn Python programming'},
        ]
        result = self.analyzer.analyze_conversation(messages)
        assert len(result['goals']) > 0

    def test_detects_action_items(self):
        messages = [
            {'role': 'user', 'content': 'remind me to call Alice tomorrow'},
        ]
        result = self.analyzer.analyze_conversation(messages)
        assert len(result['action_items']) > 0

    def test_detects_positive_tone(self):
        messages = [
            {'role': 'user', 'content': 'This is great! I love this! Awesome work!'},
        ]
        result = self.analyzer.analyze_conversation(messages)
        assert result['tone'] == 'positive'

    def test_detects_negative_tone(self):
        messages = [
            {'role': 'user', 'content': 'This is wrong and bad. I am frustrated.'},
        ]
        result = self.analyzer.analyze_conversation(messages)
        assert result['tone'] == 'negative'

    def test_processes_last_10_messages(self):
        messages = [{'role': 'user', 'content': f'Message {i}'} for i in range(20)]
        result = self.analyzer.analyze_conversation(messages)
        assert isinstance(result, dict)
        assert 'summary' in result

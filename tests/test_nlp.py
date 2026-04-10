"""
Tests for the NLP module (nlp.py).

Covers:
- spaCy label → ontology type mapping
- spaCy-based entity extraction with real model
- Regex fallback when spaCy is unavailable
- Edge cases (empty text, unicode, long text)
- Model loading and singleton behavior
"""

import pytest
from agent_brain.nlp import (
    extract_entities_spacy,
    extract_entities_regex,
    map_spacy_label,
    get_nlp,
    reset_nlp,
    SPACY_LABEL_MAP,
    LABEL_CONFIDENCE,
    SKIP_LABELS,
)


# ============================================================================
# LABEL MAPPING
# ============================================================================

class TestLabelMapping:
    """Test spaCy label to ontology type mapping."""

    def test_person_maps_to_person(self):
        assert map_spacy_label("PERSON") == "person"

    def test_org_maps_to_organization(self):
        assert map_spacy_label("ORG") == "organization"

    def test_gpe_maps_to_place(self):
        assert map_spacy_label("GPE") == "place"

    def test_loc_maps_to_place(self):
        assert map_spacy_label("LOC") == "place"

    def test_fac_maps_to_place(self):
        assert map_spacy_label("FAC") == "place"

    def test_date_maps_to_date(self):
        assert map_spacy_label("DATE") == "date"

    def test_time_maps_to_date(self):
        assert map_spacy_label("TIME") == "date"

    def test_event_maps_to_event(self):
        assert map_spacy_label("EVENT") == "event"

    def test_product_maps_to_object(self):
        assert map_spacy_label("PRODUCT") == "object"

    def test_language_maps_to_technology(self):
        assert map_spacy_label("LANGUAGE") == "technology"

    def test_unknown_label_defaults_to_concept(self):
        assert map_spacy_label("UNKNOWN_LABEL") == "concept"
        assert map_spacy_label("") == "concept"

    def test_all_labels_have_confidence(self):
        """Every mapped label should have a confidence score."""
        for label in SPACY_LABEL_MAP:
            assert label in LABEL_CONFIDENCE, f"Missing confidence for {label}"
            assert 0.0 < LABEL_CONFIDENCE[label] <= 1.0

    def test_skip_labels_are_subset_of_mapping(self):
        """All skip labels should exist in the mapping."""
        for label in SKIP_LABELS:
            assert label in SPACY_LABEL_MAP, f"Skip label {label} not in mapping"


# ============================================================================
# SPACY EXTRACTION (requires spacy + en_core_web_sm)
# ============================================================================

@pytest.fixture(autouse=True)
def _reset_nlp_cache():
    """Reset the NLP model cache before each test."""
    reset_nlp()
    yield
    reset_nlp()


class TestSpacyExtraction:
    """Tests that require spaCy to be installed."""

    @pytest.fixture(autouse=True)
    def skip_if_no_spacy(self):
        """Skip if spaCy is not available."""
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            pytest.skip("spaCy or en_core_web_sm not available")

    def test_extracts_person(self):
        entities = extract_entities_spacy("Alice went to the store")
        names = [e["name"] for e in entities]
        types = {e["name"]: e["type"] for e in entities}
        assert "Alice" in names
        assert types["Alice"] == "person"

    def test_extracts_organization(self):
        entities = extract_entities_spacy("She works at Google")
        names = [e["name"] for e in entities]
        types = {e["name"]: e["type"] for e in entities}
        assert "Google" in names
        assert types["Google"] == "organization"

    def test_extracts_place(self):
        entities = extract_entities_spacy("He traveled to Tokyo last week")
        names = [e["name"] for e in entities]
        types = {e["name"]: e["type"] for e in entities}
        assert "Tokyo" in names
        assert types["Tokyo"] == "place"

    def test_extracts_date(self):
        entities = extract_entities_spacy("The meeting is on March 15th")
        types = [e["type"] for e in entities]
        assert "date" in types

    def test_mixed_entity_sentence(self):
        """Test the canonical example from the improvement plan."""
        entities = extract_entities_spacy(
            "Alice works at Google in Tokyo last March"
        )
        names = [e["name"] for e in entities]
        types = {e["name"]: e["type"] for e in entities}

        # Should extract multiple entity types (not all 'concept')
        unique_types = set(types.values())
        assert len(unique_types) >= 2, (
            f"Expected multiple entity types, got {unique_types}. "
            f"Entities: {entities}"
        )

        # At least person and organization should be present
        assert "Alice" in names, f"Expected 'Alice' in {names}"
        assert types.get("Alice") == "person"

    def test_deduplicates(self):
        entities = extract_entities_spacy("Alice met Alice. Alice was happy.")
        alice_count = sum(1 for e in entities if e["name"] == "Alice")
        assert alice_count == 1, "Should deduplicate entities"

    def test_respects_max_entities(self):
        # Text with many entities
        text = "Alice, Bob, Charlie, David, Eve, Frank, Grace, Helen, Ivan, Jack, Karen, Larry, Michael, Nancy, Oscar"
        entities = extract_entities_spacy(text, max_entities=5)
        assert len(entities) <= 5

    def test_skips_noisy_labels(self):
        entities = extract_entities_spacy("There are 42 items worth $100 and 75%")
        types = [e["type"] for e in entities]
        # CARDINAL (42), MONEY ($100), PERCENT (75%) should be skipped
        names = [e["name"] for e in entities]
        assert "42" not in names
        assert "75%" not in names

    def test_includes_confidence(self):
        entities = extract_entities_spacy("Alice works at Google")
        for e in entities:
            assert "confidence" in e
            assert 0.0 < e["confidence"] <= 1.0

    def test_empty_text(self):
        entities = extract_entities_spacy("")
        assert entities == []

    def test_short_text(self):
        entities = extract_entities_spacy("hi")
        assert isinstance(entities, list)

    def test_unicode_text(self):
        entities = extract_entities_spacy("François visited München and São Paulo")
        assert isinstance(entities, list)

    def test_very_long_text(self):
        # Should not crash on very long text (truncated internally)
        text = "Alice works at Google. " * 1000
        entities = extract_entities_spacy(text)
        assert isinstance(entities, list)
        assert len(entities) <= 15

    def test_quoted_phrases_supplemented(self):
        """Quoted phrases should be extracted even if spaCy doesn't catch them."""
        entities = extract_entities_spacy('The project is called "Agent Brain" and it is great')
        names = [e["name"] for e in entities]
        assert "Agent Brain" in names

    def test_entities_have_description(self):
        entities = extract_entities_spacy("Alice works at Google")
        for e in entities:
            assert "description" in e
            assert isinstance(e["description"], str)


# ============================================================================
# REGEX FALLBACK
# ============================================================================

class TestRegexFallback:
    """Tests for the regex extraction fallback."""

    def test_extracts_capitalized_words(self):
        entities = extract_entities_regex("Alice went to Paris")
        names = [e["name"] for e in entities]
        assert "Alice" in names
        assert "Paris" in names

    def test_all_typed_as_concept(self):
        entities = extract_entities_regex("Alice went to Paris")
        for e in entities:
            assert e["type"] == "concept"

    def test_extracts_quoted_phrases(self):
        entities = extract_entities_regex('He said "hello world" today')
        names = [e["name"] for e in entities]
        assert "hello world" in names

    def test_deduplicates(self):
        entities = extract_entities_regex("Alice met Alice again")
        alice_count = sum(1 for e in entities if e["name"] == "Alice")
        assert alice_count == 1

    def test_empty_text(self):
        entities = extract_entities_regex("")
        assert entities == []

    def test_no_entities(self):
        entities = extract_entities_regex("this is all lowercase with no quotes")
        assert entities == []

    def test_respects_max(self):
        text = " ".join([f"Entity{i}" for i in range(20)])
        entities = extract_entities_regex(text, max_entities=5)
        assert len(entities) <= 5

    def test_lower_confidence_than_spacy(self):
        entities = extract_entities_regex("Alice went to Paris")
        for e in entities:
            assert e["confidence"] <= 0.5


# ============================================================================
# MODEL LOADING
# ============================================================================

class TestModelLoading:
    def test_get_nlp_returns_model_or_none(self):
        nlp = get_nlp()
        if nlp is not None:
            # If spaCy is available, verify it's a Language model
            import spacy
            assert isinstance(nlp, spacy.language.Language)

    def test_singleton_returns_same_object(self):
        nlp1 = get_nlp()
        nlp2 = get_nlp()
        if nlp1 is not None:
            assert nlp1 is nlp2, "Should return the same cached model"

    def test_reset_clears_cache(self):
        get_nlp()
        reset_nlp()
        # After reset, the internal state should be cleared
        from agent_brain.nlp import _nlp_model, _spacy_available
        assert _nlp_model is None
        assert _spacy_available is None

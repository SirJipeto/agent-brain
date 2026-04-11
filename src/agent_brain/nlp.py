"""
Shared NLP utilities for agent_brain.

Provides:
- Lazy-loaded spaCy model singleton
- spaCy label → ontology type mapping
- spaCy-based entity extraction function
- Regex fallback for when spaCy is unavailable
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# SPACY LABEL → ONTOLOGY TYPE MAPPING
# ============================================================================

# Maps spaCy NER labels to agent_brain ontology types (from ONTOLOGY.md)
SPACY_LABEL_MAP: Dict[str, str] = {
    # People
    "PERSON": "person",

    # Organizations
    "ORG": "organization",

    # Places
    "GPE": "place",          # Countries, cities, states
    "LOC": "place",          # Non-GPE locations (mountains, bodies of water)
    "FAC": "place",          # Facilities (airports, highways, bridges)

    # Temporal
    "DATE": "date",
    "TIME": "date",

    # Events & works
    "EVENT": "event",
    "WORK_OF_ART": "concept",
    "LAW": "concept",

    # Objects & products
    "PRODUCT": "object",

    # Groups & languages
    "NORP": "concept",       # Nationalities, religious/political groups
    "LANGUAGE": "technology",

    # Numeric (generally less useful for graph memory, but still captured)
    "MONEY": "concept",
    "PERCENT": "concept",
    "QUANTITY": "concept",
    "ORDINAL": "concept",
    "CARDINAL": "concept",
}

# Confidence scores by spaCy label (higher = more reliable NER category)
LABEL_CONFIDENCE: Dict[str, float] = {
    "PERSON": 0.90,
    "ORG": 0.85,
    "GPE": 0.90,
    "LOC": 0.80,
    "FAC": 0.75,
    "DATE": 0.85,
    "TIME": 0.80,
    "EVENT": 0.75,
    "PRODUCT": 0.70,
    "WORK_OF_ART": 0.65,
    "NORP": 0.70,
    "LANGUAGE": 0.80,
    "LAW": 0.70,
    "MONEY": 0.85,
    "PERCENT": 0.85,
    "QUANTITY": 0.75,
    "ORDINAL": 0.70,
    "CARDINAL": 0.70,
}

# Labels to skip entirely (too noisy for graph memory)
SKIP_LABELS = {"CARDINAL", "ORDINAL", "PERCENT", "QUANTITY"}


# ============================================================================
# SPACY MODEL LOADER (SINGLETON)
# ============================================================================

_nlp_model = None
_spacy_available: Optional[bool] = None


def _check_spacy_available() -> bool:
    """Check if spaCy and a language model are available."""
    global _spacy_available
    if _spacy_available is not None:
        return _spacy_available

    try:
        import spacy
        spacy.load("en_core_web_sm")
        _spacy_available = True
    except (ImportError, OSError):
        try:
            import spacy
            import subprocess
            logger.info("Attempting to auto-download spaCy en_core_web_sm model...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            _spacy_available = True
        except Exception as e:
            _spacy_available = False
            logger.info(
                f"spaCy or en_core_web_sm not available. Auto-download failed: {e}. "
                "Install manually with: pip install spacy && python -m spacy download en_core_web_sm"
            )
    return _spacy_available


def get_nlp():
    """
    Get the shared spaCy NLP model (lazy-loaded singleton).

    Returns None if spaCy is not installed or the model is missing.
    """
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model

    if not _check_spacy_available():
        return None

    import spacy
    logger.info("Loading spaCy model 'en_core_web_sm'...")
    _nlp_model = spacy.load("en_core_web_sm")
    return _nlp_model


def reset_nlp():
    """Reset the cached model (useful for testing)."""
    global _nlp_model, _spacy_available
    _nlp_model = None
    _spacy_available = None


# ============================================================================
# SPACY-BASED ENTITY EXTRACTION
# ============================================================================

def extract_entities_spacy(text: str, max_entities: int = 15) -> List[Dict]:
    """
    Extract entities from text using spaCy NER.

    Returns a list of entity dicts compatible with Neo4jBrain:
        [{"name": "Alice", "type": "person", "description": "...", "confidence": 0.9}, ...]

    Falls back to regex extraction if spaCy is unavailable.
    """
    nlp = get_nlp()
    if nlp is None:
        return extract_entities_regex(text, max_entities)

    # Process text (truncate to avoid spaCy memory issues on very long text)
    doc = nlp(text[:10000])

    entities = []
    seen = set()

    for ent in doc.ents:
        # Skip noisy labels
        if ent.label_ in SKIP_LABELS:
            continue

        # Clean entity name
        name = ent.text.strip()
        if not name or len(name) <= 1:
            continue

        # Deduplicate (case-insensitive)
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)

        # Map label to ontology type
        entity_type = SPACY_LABEL_MAP.get(ent.label_, "concept")
        confidence = LABEL_CONFIDENCE.get(ent.label_, 0.5)

        entities.append({
            "name": name,
            "type": entity_type,
            "description": f"{ent.label_} entity detected by NER",
            "confidence": confidence,
        })

        if len(entities) >= max_entities:
            break

    # Supplement with quoted phrases (spaCy doesn't catch these)
    quotes = re.findall(r'"([^"]+)"', text)
    for quote in quotes[:3]:
        key = quote.lower().strip()
        if key not in seen and len(quote) > 3:
            seen.add(key)
            entities.append({
                "name": quote.strip(),
                "type": "concept",
                "description": "Quoted phrase",
                "confidence": 0.4,
            })

    return entities[:max_entities]


# ============================================================================
# REGEX FALLBACK
# ============================================================================

def extract_entities_regex(text: str, max_entities: int = 10) -> List[Dict]:
    """
    Regex-based entity extraction fallback.

    Used when spaCy is not available. All entities typed as 'concept'
    since regex cannot classify entity types.
    """
    entities = []
    seen = set()

    # Capitalized word sequences (proper nouns)
    # Fixed regex: requires each word to start with uppercase
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    for phrase in capitalized[:max_entities]:
        key = phrase.lower()
        if key not in seen and len(phrase) > 2:
            seen.add(key)
            entities.append({
                "name": phrase,
                "type": "concept",
                "description": "Extracted from text (regex)",
                "confidence": 0.4,
            })

    # Quoted phrases
    quotes = re.findall(r'"([^"]+)"', text)
    for quote in quotes[:3]:
        key = quote.lower().strip()
        if key not in seen and len(quote) > 3:
            seen.add(key)
            entities.append({
                "name": quote.strip(),
                "type": "concept",
                "description": "Quoted phrase",
                "confidence": 0.3,
            })

    return entities[:max_entities]


def map_spacy_label(label: str) -> str:
    """Map a spaCy NER label to an ontology entity type."""
    return SPACY_LABEL_MAP.get(label, "concept")

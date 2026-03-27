# Personal AI Brain Ontology

## Overview

This ontology defines the node and relationship schema for an associative memory system that enables:
- Multi-hop relational reasoning
- Proactive memory surfacing
- Hybrid vector + graph search
- Temporal context tracking

---

## 🟦 Node Types

### :Person
Human users and contacts.

```cypher
(:Person {
  id: string,                    // unique identifier (e.g., "user", "alice")
  name: string,                  // display name
  goals: list<string>,           // stated goals: ["learn Rust", "launch startup"]
  preferences: map,              // {priority: "high", communication: "direct"}
  traits: list<string>,          // personality/contextual traits
  created_at: datetime,
  updated_at: datetime
})
```

### :Entity
Concrete, named things in the world.

```cypher
(:Entity {
  id: string,                    // stable identifier (UUID or slug)
  name: string,                  // canonical name
  type: enum[                    // classifies the entity
    person,                      // named individual
    place,                       // location
    object,                      // physical thing
    organization,                // company, team, institution
    concept,                     // abstract idea
    event,                       // something that happened
    project,                     // work endeavor
    document,                    // file, article, resource
    technology                   // software, hardware, framework
  ],
  description: string,           // human-readable summary
  importance: float,             // 0.0-1.0, affects surfacing priority
  embedding: vector[384|768],    // native Neo4j vector for semantic search
  tags: list<string>,            // lightweight categorization
  created_at: datetime,
  updated_at: datetime
})
```

### :Concept
Abstract knowledge units and domains.

```cypher
(:Concept {
  id: string,
  name: string,                  // e.g., "machine learning", "productivity"
  category: enum[
    skill,                       // learnable capability
    domain,                      // knowledge area
    topic,                       // subject matter
    technology,                  // tech stack/component
    value,                       // principle/belief
    methodology                  // process/approach
  ],
  description: string,
  embedding: vector[384|768],
  related_terms: list<string>,   // synonyms/closely related
  created_at: datetime
})
```

### :Memory
Episodic records of conversations, events, or documents.

```cypher
(:Memory {
  id: string,
  content: string,                // full text or summary
  content_type: enum[
    conversation,                 // dialogue transcript
    document,                      // parsed file content
    extracted,                     // LLM-extracted facts
    reflection,                     // AI-generated insight
    note                           // user-provided snippet
  ],
  summary: string,                // concise summary for quick recall
  embedding: vector[384|768],
  importance: float,               // 0.0-1.0, set explicitly or inferred
  salience_tags: list<string>,    // ["goal", "preference", "fact"]
  container: string,               // grouping context ("work", "personal", project name)
  timestamp: datetime,             // when the memory occurred
  expires_at: datetime,            // optional TTL for transient memories
  archived: boolean,               // true = consolidated, embeddings may be stripped
  source: string,                  // "user_message", "document:/path/file.pdf", "ai_reflection"
  metadata: map,                   // flexible key-value extras
  created_at: datetime,
  updated_at: datetime
})
```

### :Fact
Structured, attributed knowledge triples.

```cypher
(:Fact {
  id: string,
  subject: string,               // entity or concept name
  predicate: string,              // relationship verb
  object: string,                 // value or target
  confidence: float,              // 0.0-1.0, LLM-assigned certainty
  source_memory: string,          // memory ID this was extracted from
  valid_from: datetime,
  valid_until: datetime,          // null = currently valid
  superseded_by: string,          // fact ID that replaced this
  is_current: boolean,
  metadata: map,
  created_at: datetime
})
```

---

## 🔗 Relationship Types

### Knowledge & Understanding

```cypher
// Person understands a concept with given depth
(:Person)-[:KNOWS_ABOUT {
  strength: float,                // 0.0-1.0, how well understood
  confidence: float,              // certainty of this knowledge
  last_reinforced: datetime       // when this was confirmed/updated
}]->(:Concept)

// Person has interest/preference
(:Person)-[:INTERESTED_IN {
  strength: float,                 // interest level
  recency: datetime,               // last mentioned/confirmed
  context: string                  // in what context
}]->(:Concept|:Entity)

// Person is working on a project
(:Person)-[:WORKS_ON {
  role: string,                   // "lead", "contributor", "stakeholder"
  since: datetime,
  status: enum[active, paused, completed]
}]->(:Entity)
```

### Entity Relationships

```cypher
// Generic relatedness with contextual meaning
(:Entity)-[:RELATED_TO {
  weight: float,                  // 0.0-1.0, connection strength
  context: string,                // why related: "similar_to", "part_of", "used_with"
  bidirectional: boolean          // true = applies both directions
}]->(:Entity)

// Hierarchical containment
(:Entity)-[:PART_OF {
  role: string                    // "contains", "belongs_to", "contains"
}]->(:Entity)

// Causal/dependency chain
(:Entity)-[:CAUSED_BY]->(:Entity)
(:Entity)-[:ENABLES]->(:Entity)
(:Entity)-[:DEPENDS_ON]->(:Entity)

// Spatial/locational
(:Entity)-[:LOCATED_AT]->(:Entity:Place)
(:Entity)-[:NEAR]->(:Entity)
```

### Concept Relationships

```cypher
// Conceptual hierarchy
(:Concept)-[:IS_A]->(:Concept)     // inheritance
(:Concept)-[:PART_OF]->(:Concept)   // domain membership

// Implication chain (reasoning)
(:Concept)-[:IMPLIES]->(:Concept)
(:Concept)-[:CONTRADICTS]->(:Concept)

// Skill progression
(:Concept)-[:PREREQUISITE_FOR]->(:Concept)
(:Concept)-[:LEADS_TO]->(:Concept)
```

### Memory Connections

```cypher
// Memory mentions an entity
(:Memory)-[:MENTIONS {
  relevance: float,              // how central to the memory
  context: string                 // how it is mentioned
}]->(:Entity|:Concept|:Person)

// Memory is thematically about
(:Memory)-[:ABOUT]->(:Entity|:Concept)

// Temporal ordering
(:Memory)-[:BEFORE]->(:Memory)     // this happened before
(:Memory)-[:AFTER]->(:Memory)      // this happened after
(:Memory)-[:FOLLOWS {sequence:int}]->(:Memory)  // explicit ordering

// Explicit relevance (user/AI tagged)
(:Memory)-[:RELEVANT_TO {
  relevance_score: float,
  reason: string
}]->(:Memory)
```

### Fact Relationships

```cypher
(:Fact)-[:EXTRACTED_FROM]->(:Memory)
(:Fact)-[:ATTRIBUTED_TO]->(:Person)   // who confirmed this
(:Fact)-[:SUPPORTS]->(:Fact)
(:Fact)-[:CONTRADICTS]->(:Fact)
```

---

## 📊 Property Guidelines

### Embeddings
- **Dimensions**: 384 (default, faster) or 768 (more expressive)
- **Similarity function**: cosine (recommended for semantic search)
- **Index type**: vector index for ANN search

### Weights & Scores
- **Range**: 0.0 - 1.0 (normalized)
- **Decay**: Weak connections (<0.2) should be pruned periodically
- **Boosting**: Recent interactions increase weight temporarily

### Temporal Properties
- All timestamps: ISO 8601 format
- `valid_from/until`: For time-bounded facts
- `last_reinforced`: Updates on mention/confirmation

### Importance & Salience
- **importance**: General significance (set by user or inferred)
- **salience_tags**: ["goal", "preference", "deadline", "personal", "sensitive"]
- High-salience memories are surfaced more readily

---

## 🔄 Example Graph Fragments

### Travel Planning Context

```cypher
(flight:Entity {type:"event", name:"Flight to Tokyo"})
  -[:DEPENDS_ON]->(visa:Entity {type:"object", name:"Visa"})
  -[:LOCATED_AT]->(japan:Entity {type:"place", name:"Japan"})

(user:Person {name:"user"})
  -[:WORKS_ON]->(flight)
  -[:INTERESTED_IN {context:"packing tips"}]->(japan)

(packing_list:Memory {content:"Remember to pack..."})
  -[:ABOUT]->(flight)
  -[:MENTIONS]->(visa)
```

### Learning Context

```cypher
(python:Concept {name:"Python", category:"technology"})
  -[:PREREQUISITE_FOR]->(ml:Concept {name:"Machine Learning", category:"domain"})
  -[:RELATED_TO {context:"used_for"}]->(data_science:Concept {category:"domain"})

(user:Person {name:"user"})
  -[:KNOWS_ABOUT {strength:0.8, confidence:0.9}]->(python)
  -[:KNOWS_ABOUT {strength:0.3, confidence:0.6}]->(ml)
  -[:INTERESTED_IN {strength:0.9}]->(ml)

(learning_memory:Memory {source:"conversation"})
  -[:ABOUT]->(ml)
  -[:MENTIONS]->(python)
```

---

## 🚀 Implementation Notes

1. **Indexing**: Create vector indexes on `embedding` properties for all searchable nodes
2. **Constraints**: Add unique constraints on `id` for all node types
3. **Labels**: Use label-based partitioning for efficient queries
4. **Migrations**: Version this file; schema changes require migration scripts
5. **Validation**: LLM-extracted data should be validated against this schema

---

## Version
- **v1.0** - Initial ontology
- **Date**: 2024-01-01

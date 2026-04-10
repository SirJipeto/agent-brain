"""
Example FastAPI service wrapping the agent_brain Neo4jBrain.
Demonstrates startup lifecycle, dependency injection, and health checks.

Usage:
  pip install fastapi uvicorn
  uvicorn examples.fastapi_service:app --reload
"""
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

from agent_brain.brain import Neo4jBrain

# Global brain instance
brain: Optional[Neo4jBrain] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global brain
    # Initialize connection and warm up embeddings at startup
    brain = Neo4jBrain(preload_embeddings=True)
    yield
    # Cleanup on shutdown
    if brain and brain.conn:
        brain.conn.close()

app = FastAPI(title="Agent Brain API", lifespan=lifespan)

# Dependency to get brain
def get_brain() -> Neo4jBrain:
    if brain is None:
        raise HTTPException(status_code=500, detail="Brain not initialized")
    return brain

class MemoryRequest(BaseModel):
    content: str
    container: Optional[str] = "default"

class SearchRequest(BaseModel):
    query: str
    container: Optional[str] = None
    top_k: int = 5

@app.get("/health")
def health_check(b: Neo4jBrain = Depends(get_brain)):
    """Returns database connectivity, node sizing, and open-telemetry health mapping."""
    return b.get_health()

@app.post("/memory")
def add_memory(req: MemoryRequest, b: Neo4jBrain = Depends(get_brain)):
    """Ingest a new memory, extract entities, and weave it into the associative graph."""
    mem_id = b.add_memory(content=req.content, container=req.container)
    return {"status": "success", "memory_id": mem_id}

@app.post("/search")
def search_memories(req: SearchRequest, b: Neo4jBrain = Depends(get_brain)) -> Dict[str, Any]:
    """Execute a Hybrid Search (semantic vector matching + GraphRAG activation spreading)."""
    results = b.hybrid_search(query_str=req.query, top_k=req.top_k, container=req.container)
    return results

if __name__ == "__main__":
    import uvicorn
    print("Starting Agent Brain server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

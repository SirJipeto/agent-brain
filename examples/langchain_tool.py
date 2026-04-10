"""
Example of wrapping agent_brain as a LangChain/LangGraph Tool.

Usage:
  pip install langchain-core
"""
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from agent_brain.brain import Neo4jBrain

class BrainSearchInput(BaseModel):
    query: str = Field(description="The question or topic you want to remember or search for.")
    container: Optional[str] = Field(default=None, description="Optional logical partition, e.g., 'user_123'.")

class AgentBrainSearchTool(BaseTool):
    name: str = "agent_brain_search"
    description: str = "Queries the associative memory graph for past episodic memories and contextual facts."
    args_schema: Type[BaseModel] = BrainSearchInput
    
    # We hide the brain instance from pydantic fields to prevent serialization issues
    _brain: Neo4jBrain = None
    
    def __init__(self, brain_instance: Neo4jBrain, **kwargs):
        super().__init__(**kwargs)
        self._brain = brain_instance

    def _run(self, query: str, container: Optional[str] = None) -> str:
        """Execute the search and return a formatted string for the LLM."""
        try:
            results = self._brain.hybrid_search(query_str=query, container=container)
            
            # Format results concisely for the LLM
            formatted = []
            
            # Add semantic memories
            semantic = results.get("semantic_matches", [])
            if semantic:
                formatted.append("--- Precise Memories ---")
                for s in semantic:
                    formatted.append(f"- {s.get('content')}")
            
            # Add graph insights (spread activation context)
            graph = results.get("graph_insights", [])
            if graph:
                formatted.append("--- Associated Concepts & Facts ---")
                for g in graph:
                    # Ignore the raw embeddings and just output content
                    formatted.append(f"- Mentioned: {g.get('content')}")
            
            if not formatted:
                return "No relevant memories found."
                
            return "\n".join(formatted)
            
        except Exception as e:
            return f"Error accessing memory: {str(e)}"

# Usage Example
if __name__ == "__main__":
    # 1. Initialize the brain
    # (assuming Neo4j is running at localhost:7687 and environment variables are set)
    print("Connecting to brain...")
    try:
        brain = Neo4jBrain()
        
        # 2. Create the tool
        search_tool = AgentBrainSearchTool(brain_instance=brain)
        
        # 3. Use it manually or pass to a LangChain agent
        print("Testing tool manually. Querying: 'What is Python?'")
        result = search_tool.invoke({"query": "What is Python?", "container": "default"})
        
        print("\n--- LLM Context Output ---")
        print(result)
        
    except Exception as e:
        print(f"Failed to start example (Is Neo4j running?): {e}")

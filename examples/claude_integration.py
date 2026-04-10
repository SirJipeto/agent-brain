"""
Example of integrating agent_brain directly with Anthropic's Claude API.

Usage:
  pip install anthropic agent-brain
  export ANTHROPIC_API_KEY="sk-ant-..."
  python examples/claude_integration.py
"""
import os
from agent_brain.brain import Neo4jBrain

def main():
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Please run: pip install anthropic")
        return

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable.")
        return

    client = Anthropic()
    
    # Initialize the associative memory
    print("Connecting to Neo4j and initializing Agent Brain...")
    try:
        brain = Neo4jBrain(preload_embeddings=True)
    except Exception as e:
        print(f"Failed to connect to Brain: {e}")
        return
        
    # Let's seed a fake memory for testing
    print("Seeding a recent memory into the graph...")
    brain.add_memory(
        "I absolutely love fresh sushi, but my doctor told me I need to avoid high mercury fish like tuna.",
        container="user_alice"
    )
    
    user_message = "Where should we go out for dinner tonight?"
    
    # 1. Ask Brain for context BEFORE prompting Claude
    print(f"\nQuerying Brain for context related to: '{user_message}'")
    memory_results = brain.hybrid_search(query_str=user_message, container="user_alice")
    
    # Flatten the retrieved memories into a context string
    context_str = ""
    for mem in memory_results.get("semantic_matches", []):
        context_str += f"- {mem.get('content')}\n"
    for mem in memory_results.get("graph_insights", []):
        context_str += f"- (Implicit Insight) {mem.get('content')}\n"
        
    print(f"\nConstructed Memory Context:\n{context_str or 'None'}")
    
    # 2. Construct the prompt with the injected memory
    prompt = f"""You are a helpful AI assistant with access to a long-term associative memory.
    
Here are relevant facts from the user's past memories:
<memory>
{context_str}
</memory>

User Request: {user_message}

Please respond to the request, using the memory context if it is relevant.
"""

    print("\nAsking Claude...")
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    print("\nClaude's Response:")
    print("=" * 40)
    print(message.content[0].text)
    print("=" * 40)
    
    # 3. (Optional) Save Claude's response back to memory
    print("\nSaving conversation to memory...")
    brain.add_memory(
        content=f"User: {user_message}\nAI: {message.content[0].text}",
        content_type="conversation",
        container="user_alice"
    )
    
    print("Run complete!")

if __name__ == "__main__":
    main()

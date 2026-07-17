# LangGraph for Stateful AI Workflows

LangGraph is a library built on top of LangChain to create stateful, multi-actor applications with LLMs. It lets you model your agent workflows as graphs (nodes and edges).

## Why LangGraph?

While LangChain's standard chains (LCEL) are DAGs (Directed Acyclic Graphs), LangGraph allows **cycles**. This is crucial for agents that need to loop, self-correct, or retry actions until a condition is met.

## Key Components

- **State**: A typed dictionary or Pydantic model that is passed around and updated by nodes.
- **Nodes**: Python functions that receive the current state, perform an action, and return updates to the state.
- **Edges**: Conditional logic that dictates which node should run next based on the current state.

## Example: Self-Corrective RAG

Here is how you might structure a self-corrective RAG pipeline in LangGraph. If the `grade_document` node determines the document is irrelevant, a conditional edge loops back to `rewrite_query`.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class RAGState(TypedDict):
    query: str
    documents: list
    answer: str

def retrieve(state):
    # Retrieve documents
    return {"documents": ["Doc1", "Doc2"]}

def generate(state):
    # Generate answer
    return {"answer": "Here is the answer..."}

graph = StateGraph(RAGState)
graph.add_node("retrieval", retrieve)
graph.add_node("generation", generate)

graph.set_entry_point("retrieval")
graph.add_edge("retrieval", "generation")
graph.add_edge("generation", END)

app = graph.compile()
```

## State Management

Every node must return a dictionary representing the *updates* to the state, rather than the entire state itself. The underlying LangGraph engine handles merging the updates into the global state.

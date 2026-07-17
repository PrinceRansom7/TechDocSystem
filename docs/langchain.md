# Introduction to LangChain

LangChain is a framework for developing applications powered by language models. It provides standard, extendable interfaces and external integrations for building robust LLM apps.

## Core Concepts

LangChain's architecture revolves around several key concepts:

1. **Prompt Templates**: Reusable templates for generating prompts dynamically.
2. **LLMs and Chat Models**: Standard interfaces for interacting with models like Groq, OpenAI, or Anthropic.
3. **Chains**: Sequences of calls to LLMs and other tools.
4. **Agents**: LLMs that act as reasoning engines to decide what actions to take.
5. **Memory**: Storing state between interactions in a conversation.

## Example: Building a simple Chain

You can easily compose a prompt template and a chat model using LangChain's LCEL (LangChain Expression Language).

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatGroq(model="llama-3.1-8b-instant")

chain = prompt | model

response = chain.invoke({"topic": "bears"})
print(response.content)
```

## Vector Stores and Retrievers

LangChain integrates with dozens of vector stores like ChromaDB and Pinecone. A `Retriever` interface is provided so you can fetch context dynamically:

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("What is LangChain?")
```

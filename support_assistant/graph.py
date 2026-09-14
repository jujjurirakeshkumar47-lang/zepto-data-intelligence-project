from typing import TypedDict
from pathlib import Path
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float

base_path = Path(__file__).parent
client = chromadb.PersistentClient(path=str(base_path / "chroma_db"))
collection = client.get_collection("zepto_policies")
model = SentenceTransformer("all-MiniLM-L6-v2")

MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"

def classify_intent(state: State):
    query = state["query"].lower()
    keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    intent = "policy_question" if any(word in query for word in keywords) else "general_question"

    return {"intent": intent}

def retrieve_and_answer(state: State):
    embedding = model.encode([state["query"]]).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=3
    )

    documents = results["documents"][0]
    sources = results["ids"][0]

    if MOCK_LLM:
        snippet = documents[0][:200]
        answer = f"Based on the retrieved context: {snippet}"
        confidence = 1.0
    else:
        answer = f"Based on the retrieved context: {documents[0]}"
        confidence = 1.0

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }

def direct_answer(state: State):
    return {
        "answer": "I can only answer questions about Zepto policies right now.",
        "sources": [],
        "confidence": 1.0
    }

def route(state: State):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

graph = StateGraph(State)

graph.add_node("classify_intent", classify_intent)
graph.add_node("retrieve_and_answer", retrieve_and_answer)
graph.add_node("direct_answer", direct_answer)

graph.add_edge(START, "classify_intent")

graph.add_conditional_edges(
    "classify_intent",
    route,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

graph.add_edge("retrieve_and_answer", END)
graph.add_edge("direct_answer", END)

app = graph.compile()
"""Stage 3 + 4 of the RAG pipeline: retrieval and generation, orchestrated by LangGraph.

    START -> classify_intent --(policy_question)--> retrieve_and_answer -> END
                             \\-(general_question)-> direct_answer -------> END

Only the *generation* step inside each node branches on MOCK_LLM.
Retrieval (embedding the query + ChromaDB search) always runs for real.
"""
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app import llm
from app.config import TOP_K, mock_llm
from app.ingest import embed, get_collection
from app.prompts import ANSWER_PROMPT, CLASSIFY_PROMPT, DIRECT_PROMPT, format_context
from app.schemas import AskResponse

POLICY_KEYWORDS = ["delivery", "return", "refund", "membership", "tracking",
                   "cancel", "gift card", "support hours"]
GENERAL_ANSWER = "I can only answer questions about Zepto policies right now."
SNIPPET_CHARS = 200
MOCK_CONFIDENCE = 1.0


class AssistantState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved: list[dict]      # [{"id", "text", "similarity"}], best match first
    answer: str
    sources: list[str]
    confidence: float


# ---------- node 1: classify_intent ----------
def keyword_intent(query: str) -> str:
    q = query.lower()
    return "policy_question" if any(k in q for k in POLICY_KEYWORDS) else "general_question"


def classify_intent(state: AssistantState) -> AssistantState:
    if mock_llm():
        intent = keyword_intent(state["query"])           # graded baseline: no LLM call
    else:
        reply = llm.chat([{"role": "user", "content": CLASSIFY_PROMPT.format(question=state["query"])}])
        intent = "policy_question" if "policy_question" in reply else "general_question"
    return {"intent": intent}


def route(state: AssistantState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------- node 2: retrieve_and_answer ----------
def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Embed the query and fetch the k most similar chunks (cosine) from ChromaDB."""
    res = get_collection().query(query_embeddings=embed([query]), n_results=k,
                                 include=["documents", "distances"])
    return [{"id": cid, "text": doc, "similarity": round(1 - dist, 4)}   # cosine distance -> similarity
            for cid, doc, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0])]


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def retrieve_and_answer(state: AssistantState) -> AssistantState:
    chunks = retrieve(state["query"])                     # runs for real in both modes
    if mock_llm():
        answer = AskResponse(
            answer=f"Based on the retrieved context: {snippet(chunks[0]['text'])}",
            sources=[c["id"] for c in chunks],
            confidence=MOCK_CONFIDENCE,
        )
    else:
        prompt = ANSWER_PROMPT.format(context=format_context(chunks), question=state["query"])
        answer = llm.generate_validated(prompt)
    return {"retrieved": chunks, **answer.model_dump()}


# ---------- node 3: direct_answer ----------
def direct_answer(state: AssistantState) -> AssistantState:
    if mock_llm():
        answer = AskResponse(answer=GENERAL_ANSWER, sources=[], confidence=MOCK_CONFIDENCE)
    else:
        text = llm.chat([{"role": "user", "content": DIRECT_PROMPT.format(question=state["query"])}])
        answer = AskResponse(answer=text.strip(), sources=[], confidence=0.5)
    return {"retrieved": [], **answer.model_dump()}


def build_graph():
    g = StateGraph(AssistantState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("retrieve_and_answer", retrieve_and_answer)
    g.add_node("direct_answer", direct_answer)
    g.add_edge(START, "classify_intent")
    g.add_conditional_edges("classify_intent", route,
                            {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"})
    g.add_edge("retrieve_and_answer", END)
    g.add_edge("direct_answer", END)
    return g.compile()


graph = build_graph()


def ask(query: str) -> tuple[AskResponse, AssistantState]:
    """Run the graph; return the validated response plus the full final state."""
    state = graph.invoke({"query": query})
    response = AskResponse(answer=state["answer"], sources=state["sources"], confidence=state["confidence"])
    return response, state

"""Tests for the support assistant (all run in the default mock mode).

    cd support_assistant && python -m pytest tests -q
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import graph as g
from app import llm
from main import app


@pytest.fixture(autouse=True)
def mock_mode_and_no_llm(monkeypatch):
    """Default mock mode, and fail loudly if anything tries to call the LLM."""
    monkeypatch.delenv("MOCK_LLM", raising=False)

    def forbidden(*_a, **_k):
        raise AssertionError("LLM was called in mock mode")
    monkeypatch.setattr(llm, "chat", forbidden)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_all_8_documents_indexed():
    from app.ingest import get_collection
    assert sorted(get_collection().get()["ids"]) == [f"doc_{i:02d}" for i in range(1, 9)]


@pytest.mark.parametrize("query,intent", [
    ("What is the delivery fee?", "policy_question"),
    ("How do I get a REFUND?", "policy_question"),
    ("Can I cancel my order?", "policy_question"),
    ("Do gift cards expire?", "policy_question"),
    ("What are your support hours?", "policy_question"),
    ("What's the capital of France?", "general_question"),
    ("Tell me a joke", "general_question"),
])
def test_keyword_routing(query, intent):
    assert g.keyword_intent(query) == intent


@pytest.mark.parametrize("query,expected_doc", [
    ("What is the delivery fee for small orders?", "doc_01"),
    ("How long does a refund take?", "doc_02"),
    ("How do I cancel my membership?", "doc_03"),
    ("My order tracking map is stuck", "doc_04"),
    ("Can I cancel my order after it's packed?", "doc_05"),
    ("Do I need a photo for a damaged item refund on a big order?", "doc_06"),
    ("Can I combine two gift cards?", "doc_07"),
    ("What are your support hours?", "doc_08"),
])
def test_retrieval_finds_the_right_document(query, expected_doc):
    assert g.retrieve(query)[0]["id"] == expected_doc


def test_policy_query_via_api(client):
    r = client.post("/ask", json={"query": "How long does a refund take?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"answer", "sources", "confidence"}
    assert body["answer"].startswith("Based on the retrieved context: Returns & Refunds")
    assert body["sources"][0] == "doc_02" and len(body["sources"]) == 3
    assert body["confidence"] == 1.0


def test_general_query_via_api(client):
    r = client.post("/ask", json={"query": "What's the capital of France?"})
    assert r.json() == {"answer": g.GENERAL_ANSWER, "sources": [], "confidence": 1.0}


def test_graph_routes_to_the_right_node():
    _, s = g.ask("Can I cancel my order?")
    assert s["intent"] == "policy_question" and s["retrieved"]
    _, s = g.ask("hello there")
    assert s["intent"] == "general_question" and s["retrieved"] == []


def test_empty_query_rejected(client):
    assert client.post("/ask", json={"query": ""}).status_code == 422


# ---- the optional MOCK_LLM=0 retry logic, tested with a fake LLM ----

def test_retry_succeeds_after_bad_json(monkeypatch):
    replies = iter(["not json", json.dumps({"answer": "ok", "sources": ["doc_01"], "confidence": 0.9})])
    calls = []
    monkeypatch.setattr(llm, "chat", lambda messages, **k: calls.append(messages) or next(replies))
    out = llm.generate_validated("prompt")
    assert out.answer == "ok" and len(calls) == 2
    assert "could not be parsed" in calls[1][-1]["content"]      # corrective instruction was sent


def test_retry_gives_up_after_2_extra_attempts(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "chat", lambda messages, **k: calls.append(1) or '{"answer": "x", "confidence": 7}')
    out = llm.generate_validated("prompt")
    assert len(calls) == 3                                        # 1 attempt + 2 retries
    assert out.answer.startswith("[ERROR]") and out.sources == [] and out.confidence == 0.0

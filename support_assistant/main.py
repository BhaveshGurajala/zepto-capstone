"""FastAPI wrapper around the LangGraph support assistant.

    uvicorn main:app --port 7860
    curl -X POST localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "..."}'
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import mock_llm
from app.graph import ask
from app.ingest import get_collection
from app.schemas import AskRequest, AskResponse


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_collection()   # load the embedding model and build/open the ChromaDB index once, at startup
    yield


app = FastAPI(title="Zepto Support Assistant", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "mock_llm": mock_llm(), "indexed_chunks": get_collection().count()}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    response, _state = ask(req.query)
    return response

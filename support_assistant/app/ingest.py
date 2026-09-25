"""Stage 1 + 2 of the RAG pipeline: ingestion and embedding.

    load_documents()  reads docs/doc_01.txt ... doc_08.txt
    chunk_documents() one chunk per document (each is ~60-90 words, well under
                      the model's 256-token limit, so splitting would only cut
                      a policy in half)
    build_index()     embeds every chunk with all-MiniLM-L6-v2 and stores it in
                      the ChromaDB collection "zepto_policies" (cosine distance)

Run on its own to (re)build the index:  python -m app.ingest
"""
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_DIR, COLLECTION_NAME, DOCS_DIR, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def embed(texts: list[str]) -> list[list[float]]:
    # normalize_embeddings=True -> unit vectors, so cosine similarity = dot product
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def load_documents() -> list[dict]:
    docs = []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        title = text.split(":", 1)[0]
        docs.append({"doc_id": path.stem, "title": title, "text": text})
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Per-document chunking: chunk id = document id (e.g. 'doc_03')."""
    return [{"chunk_id": d["doc_id"], "doc_id": d["doc_id"], "title": d["title"], "text": d["text"]}
            for d in docs]


def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_index(reset: bool = True):
    client = get_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    chunks = chunk_documents(load_documents())
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        metadatas=[{"doc_id": c["doc_id"], "title": c["title"]} for c in chunks],
    )
    return collection


@lru_cache(maxsize=1)
def get_collection():
    """Open the collection, building it first if it's missing or incomplete."""
    client = get_client()
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
        if collection.count() == len(load_documents()):
            return collection
    except Exception:
        pass
    return build_index(reset=True)


if __name__ == "__main__":
    col = build_index(reset=True)
    print(f"Indexed {col.count()} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}")
    for cid, meta in zip(*[col.get()[k] for k in ("ids", "metadatas")]):
        print(f"  {cid}  {meta['title']}")

# Module 3 — Support Assistant (`/support_assistant`)

A small RAG service for Zepto's policy questions: **sentence-transformers + ChromaDB + LangGraph + Pydantic + FastAPI**, running fully offline by default.

## How to run

```bash
cd support_assistant
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build of torch
pip install -r requirements.txt

python -m app.ingest                           # optional: build the index now (the app also builds it on startup)
uvicorn main:app --host 0.0.0.0 --port 7860    # MOCK_LLM unset = mock mode (graded default)
python -m pytest tests -q                      # 22 tests, all in mock mode
```

The first run downloads `all-MiniLM-L6-v2` (about 90 MB) from Hugging Face and caches it. After that everything runs offline. `MOCK_LLM` is **left unset** everywhere below: that is the graded, deterministic mode, and no LLM API is ever called.

### With Docker

```bash
cd support_assistant
docker build -t zepto-support .
docker run --rm -p 7860:7860 zepto-support
# in another terminal:
curl -X POST localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "How long does a refund take?"}'
```

The `Dockerfile` uses `python:3.11-slim`. It installs CPU-only torch and the requirements, downloads the embedding model **at build time**, builds the ChromaDB index into the image, and serves `uvicorn main:app` on port 7860. `HF_HUB_OFFLINE=1` is set after the download, so the container needs no network at run time. `MOCK_LLM=1` is set in the image. To try the optional real LLM: `docker run -e MOCK_LLM=0 -e GROQ_API_KEY=... -p 7860:7860 zepto-support`.

## Example calls (recorded with `MOCK_LLM` unset)

**1. Policy question → `classify_intent` → `retrieve_and_answer`**

```bash
curl -X POST localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "How long does a refund take?"}'
```

```json
{"answer":"Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in...","sources":["doc_02","doc_06","doc_05"],"confidence":1.0}
```

"refund" is a keyword, so the query is routed to retrieval. The top chunk is `doc_02` (Returns & Refunds, cosine similarity 0.53), which is the right document. `sources` lists the ids of all 3 retrieved chunks, best first.

**2. A second policy question**

```bash
curl -X POST localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "Can I cancel my order after it has been packed?"}'
```

```json
{"answer":"Based on the retrieved context: Orders can be cancelled free of cost any time before the order status changes to 'Packed', typically within the first 2 minutes of placing the order. Once an order has been packed, it can no longer...","sources":["doc_05","doc_02","doc_06"],"confidence":1.0}
```

**3. General question → `classify_intent` → `direct_answer` (no retrieval)**

```bash
curl -X POST localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of France?"}'
```

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

No keyword matches, so the query goes straight to `direct_answer`. Nothing is embedded or retrieved, and `sources` is empty.

`GET /health` returns `{"status":"ok","mock_llm":true,"indexed_chunks":8}`.

## Architecture: the RAG pipeline, stage by stage

```
 docs/doc_01..08.txt
        │  (1) INGESTION         app/ingest.py: load_documents(), chunk_documents()
        ▼
 8 chunks (1 per document, id = "doc_01".."doc_08")
        │  (2) EMBEDDING         app/ingest.py: embed() with all-MiniLM-L6-v2 (384-dim, normalized)
        ▼
 ChromaDB PersistentClient (./chroma_db), collection "zepto_policies", cosine space
        │
 POST /ask {"query"} ──► main.py ──► LangGraph StateGraph (app/graph.py)
                                        │
                                  classify_intent ── keyword heuristic   [branches on MOCK_LLM]
                                   │            │
                     policy_question            general_question
                                   ▼            ▼
                    retrieve_and_answer      direct_answer
       (3) RETRIEVAL: embed query, top-3     fixed canned string   [branches on MOCK_LLM]
           cosine search in ChromaDB
           (always real, both modes)
       (4) GENERATION: canned template
           from top chunk      [branches on MOCK_LLM]
                                   │            │
                                   ▼            ▼
                       AskResponse (Pydantic): answer / sources / confidence ──► JSON reply
```

1. **Ingestion — `app/ingest.py`.** `load_documents()` reads the 8 files in `docs/`, which hold the policy text exactly as given in the brief. The titles are kept separately in `DOC_TITLES` and stored as metadata. `chunk_documents()` makes **one chunk per document**, and the chunk id is the file name (`doc_01` … `doc_08`). Each document is only 55–89 words, well under the model's 256-token limit and each about a single policy, so splitting them further would only separate related sentences (a fee from its threshold, say).
2. **Embedding — `app/ingest.py`.** `embed()` encodes each chunk with the local `sentence-transformers` model **`all-MiniLM-L6-v2`**, normalized to unit length. `build_index()` stores the vectors, texts and metadata (`doc_id`, `title`) in the **ChromaDB collection `zepto_policies`** (`PersistentClient` at `./chroma_db`, `hnsw:space = cosine`). `get_collection()` builds this automatically at app startup if it's missing.
3. **Retrieval — the `retrieve_and_answer` node, `retrieve()` in `app/graph.py`.** The query is embedded with the same model, and ChromaDB returns the **top 3** chunks by cosine similarity (similarity = 1 − cosine distance). This always runs for real, in both modes, because it needs no API key or network.
4. **Generation — `retrieve_and_answer` and `direct_answer` in `app/graph.py`.** In mock mode, `retrieve_and_answer` returns `"Based on the retrieved context: {snippet}"`, where the snippet is the first ~200 characters of the top chunk, cut at a word boundary. `direct_answer` returns the fixed string `"I can only answer questions about Zepto policies right now."`. Every answer is built as the Pydantic model `AskResponse` (`app/schemas.py`), which FastAPI uses as the `response_model`.

**Data flow:** the graph state is a `TypedDict` (`AssistantState`: `query → intent → retrieved → answer / sources / confidence`). Each node returns the keys it fills in. `add_conditional_edges("classify_intent", route, …)` reads `intent` and sends the state to one of the two answer nodes, and both lead to `END`. The routing itself never looks at `MOCK_LLM`.

### What `MOCK_LLM` changes

`MOCK_LLM` is read in `app/config.py`. Unset or `1` means mock mode; only an explicit `0` enables the real LLM.

| Stage / node | Default (mock, graded) | `MOCK_LLM=0` (optional extension) |
|---|---|---|
| Ingestion, embedding | same | same |
| `classify_intent` | Keyword heuristic: `delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours` → `policy_question`. No LLM call. | LLM classifies with `CLASSIFY_PROMPT` |
| Retrieval (in `retrieve_and_answer`) | same: real top-3 ChromaDB search | same |
| Generation in `retrieve_and_answer` | `"Based on the retrieved context: …"` from the top chunk. `sources` = the 3 retrieved ids. `confidence` = 1.0 | LLM answers with `ANSWER_PROMPT` using only the retrieved chunks, and returns JSON |
| `direct_answer` | Fixed canned string, `sources = []`, `confidence` = 1.0 | LLM replies with `DIRECT_PROMPT`, no retrieval |
| Validation | Built directly as `AskResponse`, so it can't fail | LLM JSON is validated against `AskResponse`. On failure it **retries up to 2 more times** with `CORRECTION_PROMPT`, then returns an `[ERROR] …` answer with `confidence = 0.0` (`app/llm.py: generate_validated`) |

The real-LLM path calls **Groq's free tier** (`llama-3.1-8b-instant`, OpenAI-compatible endpoint) with the key taken from the `GROQ_API_KEY` environment variable. It is never hard-coded. This path was not used for grading. Its retry logic is covered by two tests that use a fake LLM (`test_retry_succeeds_after_bad_json`, `test_retry_gives_up_after_2_extra_attempts`). The optional Hugging Face Spaces deployment was not attempted.

## The structured prompt (Task 2)

The full text is in `app/prompts.py` (`ANSWER_PROMPT`). It follows the **role → context → task → format → length** skeleton, with negative constraints and a few-shot example:

```text
### ROLE
You are Zepto's customer-support assistant. You answer customer questions about
Zepto's delivery, returns, membership, order and support policies.

### CONTEXT
The following policy excerpts were retrieved from Zepto's policy documents.
Each excerpt starts with its source id in square brackets.

{context}

### TASK
Answer the customer's question using ONLY the policy excerpts above.
- Do NOT answer using any information that is not present in the provided context.
- Do NOT guess, invent numbers, or rely on general knowledge about other companies.
- If the context does not contain the answer, reply exactly:
  "I don't have that information in Zepto's policy documents."

### FORMAT
Return a single JSON object and nothing else - no markdown, no code fences:
{"answer": "<your answer>", "sources": ["<source id>", ...], "confidence": <number between 0 and 1>}
- "sources" lists only the ids of the excerpts you actually used.
- "confidence" is how fully the excerpts answer the question (1.0 = fully).

### LENGTH
Keep "answer" to at most 3 sentences (about 60 words).

### EXAMPLE
Question: Is there a fee for delivery on a small order?
Excerpts:
[doc_01] Delivery Policy: ... Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee. ...
Output:
{"answer": "Yes. Orders below INR 149 have a flat INR 25 delivery fee; orders over INR 149 get free standard delivery.", "sources": ["doc_01"], "confidence": 0.95}

### QUESTION
{question}
```

## Retrieval check

`tests/test_assistant.py` asks one question aimed at each document and checks that the **top** retrieved chunk is the right one. All 8 pass:

| Query | Top chunk |
|---|---|
| What is the delivery fee for small orders? | doc_01 Delivery Policy |
| How long does a refund take? | doc_02 Returns & Refunds |
| How do I cancel my membership? | doc_03 Membership Tiers |
| My order tracking map is stuck | doc_04 Order Tracking |
| Can I cancel my order after it's packed? | doc_05 Order Cancellation Policy |
| Do I need a photo for a damaged item refund on a big order? | doc_06 Damaged or Missing Items |
| Can I combine two gift cards? | doc_07 Gift Cards |
| What are your support hours? | doc_08 Customer Support Hours |

## Known limitation of the mock classifier

The keyword list is fixed by the brief, so some real policy questions miss it. "How much does Zepto Pass+ cost?" or "Is phone support available?" contain none of the keywords and are routed to `direct_answer`, even though the corpus has the answer. In the optional `MOCK_LLM=0` mode the LLM classifier would catch these. In mock mode the fix would be a wider keyword list or a similarity threshold on retrieval, but I kept the heuristic exactly as specified.

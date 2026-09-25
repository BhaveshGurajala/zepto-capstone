# Zepto Data & AI Platform — AI/ML Capstone

One repository, three connected modules:

| Module | What it shows | Marks |
|---|---|---|
| [`/data_pipeline`](data_pipeline/README.md) | Scrape raw catalogue data → clean → convert GBP→INR → normalized SQLite → SQL + pandas | 25 |
| [`/analytics`](analytics/README.md) | Profile and clean the Titanic data, tell a visual story, then build, evaluate, tune and save a full ML pipeline | 50 |
| [`/support_assistant`](support_assistant/README.md) | A grounded GenAI support bot: local embeddings + ChromaDB + LangGraph + Pydantic + FastAPI + Docker | 25 |

Together they tell one story. The data pipeline gives analysts clean relational data. The analytics module shows how that kind of data is profiled and turned into a deployable prediction model. The support assistant puts a grounded GenAI service in front of Zepto's own policies.

## Setup

**Python 3.11+** is required (pandas 3 needs it).

This repo has **one `requirements.txt` per module plus a consolidated root `requirements.txt`** that includes all three. Use whichever you prefer:

```bash
git clone <this repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch for sentence-transformers
pip install -r requirements.txt                                      # everything
# or just one module, e.g.:  pip install -r data_pipeline/requirements.txt
```

No API keys, accounts or paid services are needed anywhere.

## Running each module end to end

### 1. Data pipeline

```bash
cd data_pipeline
python run_pipeline.py            # live scrape of books.toscrape.com -> clean -> SQLite -> 7 queries
python -m pytest tests -q         # parser + messy-row tests
```

Outputs: `data/raw_books.csv`, `data/clean_books.csv`, `data/books.db` and **`query_results.md`** (every SQL query with its output, plus the `pd.read_sql` vs `pd.merge` comparison). `python run_pipeline.py --skip-scrape` re-runs everything from the committed raw CSV without network access.

**Currency rate used: 1 GBP = 105.50 INR** (fixed, project-defined constant; no API call).

### 2. Analytics

```bash
cd analytics
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb
python predict.py                 # reload the saved pipeline, predict on raw rows
```

Or open the notebooks and "Run All" in order. Both are committed with outputs. `titanic.csv` is the offline fallback: `01_eda.ipynb` loads with `sns.load_dataset('titanic')` once and saves it immediately, and everything after that reads the CSV.

### 3. Support assistant

```bash
cd support_assistant
uvicorn main:app --host 0.0.0.0 --port 7860      # MOCK_LLM unset = graded mock mode
curl -X POST localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "How long does a refund take?"}'
python -m pytest tests -q

# or with Docker
docker build -t zepto-support . && docker run --rm -p 7860:7860 zepto-support
```

The first run downloads the open-source `all-MiniLM-L6-v2` embedding model (about 90 MB, cached afterwards). The Docker image downloads it at build time.

## Design decisions (summary)

### Data pipeline

- **Scope:** 4 full categories (Mystery, Historical Fiction, Poetry, Science Fiction), **93 books**, with pagination handled by following the `next` link. Categories are found by name from the site's sidebar, not hard-coded URLs.
- **Separate steps** (`scrape.py`, `clean.py`, `database.py`, `queries.py`) run by `run_pipeline.py`. The raw scrape is saved to CSV, so cleaning and loading can be re-run offline and every step can be inspected.
- **Messy rows:** numeric fields (`price_gbp`, `rating`) that fail to parse are **median-imputed**. Rows whose stock status, title or category can't be read are **dropped**, because a yes/no fact can't be sensibly imputed. Tests feed in broken rows to prove the pipeline doesn't crash.
- **Schema:** `categories(category_id PK, category_name UNIQUE)` ← `books(…, category_id FK)`, with `CHECK` constraints on `rating` (1–5) and `in_stock` (0/1). The database is rebuilt from scratch on each run.
- **Queries:** 7 queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, IN, two JOINs and a GROUP BY. The JOIN is reproduced with `pd.merge`, and `assert_frame_equal` proves the results are identical.

### Analytics

- **One load:** `sns.load_dataset` is called once. The raw data is saved to `titanic.csv`, and both notebooks share the cleaning rules in `cleaning.py`.
- **Missing values by threshold:** `embarked`/`embark_town` (0.22%) → drop rows. `age` (19.87%) → impute with the sex × class median. `deck` (77.22%) → `"Unknown"` as its own category, because whether a deck is recorded is itself informative (67% vs 30% survival).
- **No leakage:** the stratified split (61.6 / 38.4 class balance) happens before any statistic is learned. Imputation, one-hot encoding and scaling live in a `ColumnTransformer` inside a `Pipeline`, so they are only ever fit on training rows. SMOTE sits inside an imblearn pipeline, so it only touches the training fold.
- **Results:** the tuned Random Forest (`max_depth=8, max_features=0.5, n_estimators=400`, OOB 0.826) is deployed: test accuracy 0.820, precision 0.833, F1 0.738. Logistic Regression has the best AUC (0.861) and is the runner-up. The fare regression reaches R² 0.347 and shows clear heteroscedasticity.
- **Saved artifact:** the whole fitted pipeline (preprocessing + model) is saved with `joblib.dump`, so it predicts directly on raw rows, including missing ages.

### Support assistant

- **Offline by default:** every LLM step is behind `MOCK_LLM` (unset = mock). Embeddings (`all-MiniLM-L6-v2`) and ChromaDB run locally, so retrieval is real in both modes.
- **Per-document chunking:** each policy is short (55–89 words) and about one topic, so one chunk per document keeps each policy intact. Chunk id = document id, which makes `sources` easy to read.
- **LangGraph:** a `TypedDict` state, three nodes (`classify_intent`, `retrieve_and_answer`, `direct_answer`) and a conditional edge that routes on the intent, independent of `MOCK_LLM`.
- **Guaranteed schema:** every response is a Pydantic `AskResponse(answer, sources, confidence)`. The optional real-LLM path validates the LLM's JSON and retries twice with a corrective prompt before returning a marked error.
- **Docker:** CPU-only torch, with the model and index baked in at build time, so the container serves `/ask` with no network.

## Git workflow

Each module was built on its own feature branch (`feature/data-pipeline`, `feature/analytics`, `feature/support-assistant`). Each branch has at least two commits and was merged into `main` with a merge commit (`--no-ff`). `git log --graph --all --oneline` shows this.

## Repository layout

```
.
├── README.md                 ← this file
├── requirements.txt          ← consolidated (includes the three below)
├── data_pipeline/
│   ├── scrape.py  clean.py  database.py  queries.py  run_pipeline.py
│   ├── query_results.md      ← all SQL queries + outputs
│   ├── data/                 ← raw_books.csv, clean_books.csv, books.db
│   ├── tests/  requirements.txt  README.md
├── analytics/
│   ├── 01_eda.ipynb  02_modeling.ipynb  cleaning.py  predict.py
│   ├── titanic.csv           ← the one offline copy of the dataset
│   ├── models/titanic_best_pipeline.joblib
│   ├── charts/               ← PNG copies of every chart
│   ├── requirements.txt  README.md   ← all written interpretations
└── support_assistant/
    ├── docs/doc_01..08.txt   ← the policy corpus
    ├── app/  (config, ingest, prompts, schemas, llm, graph)
    ├── main.py  Dockerfile  requirements.txt  README.md
    └── tests/
```

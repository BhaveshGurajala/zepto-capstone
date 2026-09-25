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

- **Scope:** 4 full categories (Mystery, Historical Fiction, Poetry, Science Fiction), **93 books**. The scraper follows the `next` link to get every page, and finds categories by name from the site's sidebar instead of hard-coded URLs.
- **Separate steps** (`scrape.py`, `clean.py`, `database.py`, `queries.py`) run by `run_pipeline.py`. The raw scrape is saved to CSV, so cleaning and loading can be re-run offline.
- **Messy rows:** when a value can't be read, the pipeline either fills it with the median or drops the row. Price and rating are numbers, so one bad value is filled with the median and one messy cell doesn't cost us the whole book. Stock status is a yes/no fact with no sensible middle value, so guessing would be making data up and those rows are dropped. Tests feed in broken rows to prove the rules work.
- **Schema:** `categories(category_id PK, category_name UNIQUE)` ← `books(…, category_id FK)`, with `CHECK` constraints on `rating` (1–5) and `in_stock` (0/1). The database is rebuilt from scratch on each run.
- **Queries:** 7 queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, IN, two JOINs and a GROUP BY. The JOIN is rebuilt with `pd.merge`, and `assert_frame_equal` proves both give the same result.

### Analytics

- **One load:** `sns.load_dataset` is called once, the raw data is saved to `titanic.csv`, and both notebooks share the cleaning rules in `cleaning.py`.
- **Missing values:** `embarked` (0.22%) → drop the 2 rows. `age` (19.87%) → fill with the median of the same sex and class, because different groups have very different ages. `deck` (77.22%) → too much to fill, but the missing values are not random (67% vs 30% survival), so I kept it with an "Unknown" label.
- **No leakage:** the stratified split keeps the same 61.6 / 38.4 ratio in train and test, and happens before any filling-in, encoding or scaling. All preprocessing sits inside one scikit-learn `Pipeline` with the model, so every step learns only from training rows. SMOTE also runs on the training data only.
- **Results:** I would deploy the tuned Random Forest (`max_depth=8, max_features=0.5, n_estimators=400`, OOB 0.826), because it has the highest accuracy (0.820), precision (0.833) and F1 (0.738). Logistic Regression is a strong runner-up with the best AUC (0.861). The fare regression explains about a third of fare (R² 0.347) and shows clear heteroscedasticity.
- **Saved model:** the whole fitted pipeline (preprocessing + model) is saved with `joblib.dump`, so it predicts directly on raw rows, even with a missing age.

### Support assistant

- **Offline by default:** every LLM step is behind `MOCK_LLM` (unset = mock). Embeddings (`all-MiniLM-L6-v2`) and ChromaDB run locally, so retrieval always runs for real, even in mock mode.
- **One chunk per document:** each policy is short (about 55–89 words) and covers only one policy, so splitting it further would only separate related sentences.
- **LangGraph:** a `TypedDict` state, three nodes (`classify_intent`, `retrieve_and_answer`, `direct_answer`) and a conditional edge that routes on the intent.
- **Guaranteed schema:** every answer is checked by the Pydantic model `AskResponse(answer, sources, confidence)`. The optional real-LLM path retries twice with a corrective prompt if the JSON is invalid.
- **Docker:** CPU-only torch, with the model and index built into the image, so the container serves `/ask` with no network. I built and ran it on my Mac, along with the live scrape and all tests.

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

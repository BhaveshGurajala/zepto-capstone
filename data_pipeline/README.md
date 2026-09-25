# Module 1 — Data Pipeline (`/data_pipeline`)

Scrape → clean → convert → store → query, using [books.toscrape.com](http://books.toscrape.com/).

## How to run

From the repo root, after installing `requirements.txt`:

```bash
cd data_pipeline
python run_pipeline.py                # full run: live scrape, then clean, load and query
python run_pipeline.py --skip-scrape  # same, but reuse the committed data/raw_books.csv
python -m pytest tests -q             # cleaning + parser tests
```

`--skip-scrape` is there so the rest of the pipeline can be re-run offline. The live scrape takes about 10 seconds.

## Files

| File | What it does |
|---|---|
| `scrape.py` | Task 1. Uses `requests` + `BeautifulSoup`. Reads the category links from the home page sidebar, then walks every page of each category by following the `next` link. Saves the raw rows to `data/raw_books.csv`. |
| `clean.py` | Tasks 2 and 3. Parses price, rating and availability into proper types and adds `price_inr`. |
| `database.py` | Task 4. The SQLite schema. Splits the flat data into `categories` and `books` and inserts them. |
| `queries.py` | Task 5. The 7 SQL queries. |
| `run_pipeline.py` | Runs every step in order. Writes `query_results.md` and checks `pd.read_sql` against `pd.merge` (Task 6). |
| `query_results.md` | **Every query string with its output**, plus the read_sql vs merge comparison. |
| `data/books.db` | The SQLite database. `run_pipeline.py` rebuilds it from scratch on every run. |
| `tests/test_clean.py` | Checks the parsers and the messy-row handling. |

## What was scraped

4 categories, every page of each: **Mystery (32), Historical Fiction (26), Poetry (19), Science Fiction (16) = 93 books**. That's above the 60-book minimum. Mystery and Historical Fiction each have 2 pages, so the pagination code gets exercised.

Fields captured per book: `title`, `price` (e.g. `£47.82`), `star_rating` (e.g. `Four`), `availability` (e.g. `In stock`) and `category`.

## Cleaning decisions

| Raw field | Clean column | How | If it fails to parse |
|---|---|---|---|
| `price` `"£47.82"` | `price_gbp` (float) | Regex pulls out the number, so a mis-decoded `"Â£47.82"` still works | **Median imputation** |
| `star_rating` `"Four"` | `rating` (int 1–5) | Lookup table `One..Five → 1..5` | **Median imputation** (then rounded to an int) |
| `availability` `"In stock"` | `in_stock` (bool) | Text starts with "in stock" → True, "out of stock" → False | **Row dropped** |
| `title` / `category` | — | kept as text | **Row dropped** |

Why the split:

- **Price and rating are numbers**, so one bad value can be filled with the median. The median is used instead of the mean because it isn't pulled around by very cheap or very expensive books. This way one messy cell doesn't cost us the whole book.
- **Stock status is a yes/no fact.** There is no sensible "middle" value, and guessing would mean inventing data. So those rows are dropped. The same goes for a missing title or category, since a book without them can't be stored or joined.
- Duplicate (title, category) pairs are removed.

On the live site every row parsed cleanly: 0 rows dropped and 0 values imputed. `tests/test_clean.py` feeds in deliberately broken rows (`"N/A"` price, `"Zero"` rating, `"maybe"` availability) to show these rules work and the pipeline doesn't crash.

Note: every book in these 4 categories is listed as "In stock", so `in_stock` is True for all 93 rows. The parser still handles "Out of stock" correctly (it's covered by the tests).

## Currency conversion

**Fixed project rate: 1 GBP = 105.50 INR.** This is a constant set by the project brief, not a live or historical market rate. `price_inr = round(price_gbp × 105.50, 2)`. No API call and no network access are needed. The optional live-rate lookup was not attempted.

## Database schema (two tables, PK/FK)

```sql
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL UNIQUE
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT    NOT NULL,
    price_gbp   REAL    NOT NULL,
    price_inr   REAL    NOT NULL,
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock    INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
    category_id INTEGER NOT NULL REFERENCES categories(category_id)
);
```

The category name is stored once in `categories`. Each book points to it with `category_id`, so the name isn't repeated 93 times. SQLite has no boolean type, so `in_stock` is stored as 0/1.

## SQL queries

The full SQL and output for each query are in [`query_results.md`](query_results.md).

| # | Query | Clauses |
|---|---|---|
| Q1 | Five-star books | SELECT, WHERE |
| Q2 | 10 most expensive books | ORDER BY, LIMIT |
| Q3 | Rating values that occur | DISTINCT |
| Q4 | Books priced £20–£30 | BETWEEN |
| Q5 | Books rated 4 or 5 | IN, ORDER BY, LIMIT |
| Q6 | Highly rated books with category name | **JOIN**, WHERE, ORDER BY |
| Q7 | Count, average price and rating per category | **JOIN**, GROUP BY |

## read_sql vs merge (Task 6)

All 7 query results are read into DataFrames with `pd.read_sql(sql, conn)`. Then the JOIN query (Q6) is rebuilt with pandas only: `pd.merge(books, categories, on="category_id")`, then the same filter (`rating >= 4`) and the same sort. `pd.testing.assert_frame_equal` confirms the two DataFrames are identical: 41 rows, same order, same values. The first 10 rows of each are printed side by side at the end of `query_results.md`.

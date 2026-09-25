"""Step 4 + 5 - Create the normalized SQLite schema and load the clean data."""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "books.db"

SCHEMA = """
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

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
"""


def split_tables(clean: pd.DataFrame):
    """Turn the flat cleaned DataFrame into the two normalized tables."""
    categories = (
        pd.DataFrame({"category_name": sorted(clean["category"].unique())})
        .reset_index(names="category_id")
    )
    categories["category_id"] += 1  # ids start at 1

    books = clean.merge(categories, left_on="category", right_on="category_name")
    books = books.reset_index(names="book_id")
    books["book_id"] += 1
    books["in_stock"] = books["in_stock"].astype(int)
    books = books[["book_id", "title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]]
    return categories, books


def build_database(clean: pd.DataFrame, db_path: Path = DB_PATH):
    """Recreate the database from scratch and insert both tables."""
    categories, books = split_tables(clean)
    db_path.parent.mkdir(exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO categories (category_id, category_name) VALUES (?, ?)",
            categories.itertuples(index=False, name=None),
        )
        conn.executemany(
            "INSERT INTO books (book_id, title, price_gbp, price_inr, rating, in_stock, category_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            books.itertuples(index=False, name=None),
        )
        conn.commit()
    return categories, books

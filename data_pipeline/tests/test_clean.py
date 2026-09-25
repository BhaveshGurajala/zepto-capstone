"""Checks that cleaning handles messy rows without crashing.

Run from the data_pipeline folder:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clean import GBP_TO_INR, clean_books, parse_availability, parse_price, parse_rating  # noqa: E402


def test_parsers():
    assert parse_price("£51.77") == 51.77
    assert parse_price("Â£13.99") == 13.99  # mis-decoded pound sign still works
    assert pd.isna(parse_price("free!"))
    assert parse_rating("Three") == 3
    assert pd.isna(parse_rating("Six"))
    assert parse_availability("In stock (22 available)") is True
    assert parse_availability("Out of stock") is False
    assert parse_availability("???") is None


def test_messy_rows_are_imputed_or_dropped():
    raw = pd.DataFrame([
        {"title": "A", "price": "£10.00", "star_rating": "One", "availability": "In stock", "category": "X"},
        {"title": "B", "price": "£20.00", "star_rating": "Three", "availability": "In stock", "category": "X"},
        {"title": "C", "price": "£30.00", "star_rating": "Five", "availability": "In stock", "category": "Y"},
        {"title": "D", "price": "N/A", "star_rating": "Zero", "availability": "In stock", "category": "Y"},
        {"title": "E", "price": "£40.00", "star_rating": "Two", "availability": "maybe", "category": "Y"},
    ])
    out = clean_books(raw, verbose=False)

    assert list(out["title"]) == ["A", "B", "C", "D"]  # E dropped (unknown stock)
    d = out[out["title"] == "D"].iloc[0]
    assert d["price_gbp"] == 20.0  # median of 10, 20, 30
    assert d["rating"] == 3        # median of 1, 3, 5
    assert out["rating"].dtype.kind == "i"
    assert out["in_stock"].dtype == bool
    assert (out["price_inr"] == (out["price_gbp"] * GBP_TO_INR).round(2)).all()


# A real <article> copied from books.toscrape.com (Mystery, page 1), trimmed
REAL_ARTICLE = """
<article class="product_pod">
  <p class="star-rating Four"><i class="icon-star"></i></p>
  <h3><a href="../../../sharp-objects_997/index.html" title="Sharp Objects">Sharp Objects</a></h3>
  <div class="product_price">
    <p class="price_color">£47.82</p>
    <p class="instock availability"><i class="icon-ok"></i>
        In stock
    </p>
  </div>
</article>
"""


def test_parse_book_on_real_html():
    from bs4 import BeautifulSoup
    from scrape import parse_book

    article = BeautifulSoup(REAL_ARTICLE, "html.parser").select_one("article.product_pod")
    row = parse_book(article, "Mystery")
    assert row == {"title": "Sharp Objects", "price": "£47.82", "star_rating": "Four",
                   "availability": "In stock", "category": "Mystery"}

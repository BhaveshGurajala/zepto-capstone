"""Step 2 + 3 - Clean the raw scraped rows and convert GBP -> INR.

Cleaning rules (also explained in the README):
  * price        -> price_gbp (float). Unparseable -> filled with the median price.
  * star_rating  -> rating (int 1-5).   Unparseable -> filled with the median rating.
  * availability -> in_stock (bool).    Unparseable -> row is dropped.
  * title / category missing            -> row is dropped.
  * price_inr = price_gbp * 105.50 (fixed project rate, no API call).
"""
import re

import numpy as np
import pandas as pd

GBP_TO_INR = 105.50  # fixed, project-defined constant (not a market rate)

RATING_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def parse_price(text):
    """'£51.77' -> 51.77. Returns NaN if no number can be found."""
    if not isinstance(text, str):
        return np.nan
    match = re.search(r"\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group()) if match else np.nan


def parse_rating(text):
    """'Three' -> 3. Returns NaN for anything outside One..Five."""
    if not isinstance(text, str):
        return np.nan
    return RATING_WORDS.get(text.strip().lower(), np.nan)


def parse_availability(text):
    """'In stock (22 available)' -> True, 'Out of stock' -> False, else None."""
    if not isinstance(text, str):
        return None
    t = text.strip().lower()
    if t.startswith("out of stock"):
        return False
    if t.startswith("in stock"):
        return True
    return None


def clean_books(raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = raw.copy()

    df["price_gbp"] = df["price"].apply(parse_price)
    df["rating"] = df["star_rating"].apply(parse_rating)
    df["in_stock"] = df["availability"].apply(parse_availability)

    # Rows we can't use at all: no title, no category, or unknown stock status.
    # A book's stock status is a yes/no fact - guessing it would be making data up,
    # so these rows are dropped instead of imputed.
    bad = df["title"].isna() | df["category"].isna() | df["in_stock"].isna()
    if verbose:
        print(f"Dropping {int(bad.sum())} row(s) with missing title/category/availability")
    df = df[~bad].copy()

    # Numeric fields: fill a failed parse with the median, so one messy row
    # doesn't crash the pipeline or throw away an otherwise good book.
    for col in ["price_gbp", "rating"]:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            if verbose:
                print(f"Imputed {n_missing} missing '{col}' value(s) with median {median}")

    df["price_gbp"] = df["price_gbp"].astype(float).round(2)
    df["rating"] = df["rating"].round().astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    df = df.drop_duplicates(subset=["title", "category"]).reset_index(drop=True)
    return df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]

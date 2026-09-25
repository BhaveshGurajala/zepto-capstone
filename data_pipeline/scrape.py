"""Step 1 - Scrape books from books.toscrape.com.

Scrapes every book (all paginated pages) in a fixed list of categories and
writes the raw, uncleaned rows to data/raw_books.csv.

Run:  python scrape.py
"""
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/"
CATEGORIES = ["Mystery", "Historical Fiction", "Science Fiction", "Poetry"]
RAW_CSV = Path(__file__).parent / "data" / "raw_books.csv"
HEADERS = {"User-Agent": "zepto-capstone-scraper/1.0 (learning project)"}


def get_soup(session, url):
    """Download a page and return it parsed. Raises on a non-200 response."""
    resp = session.get(url, headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {url} returned HTTP {resp.status_code}")
    resp.encoding = "utf-8"  # the site is UTF-8; without this "£" becomes "Â£"
    return BeautifulSoup(resp.text, "html.parser")


def find_category_urls(session):
    """Read the sidebar on the home page and map category name -> URL."""
    soup = get_soup(session, BASE_URL)
    links = soup.select("div.side_categories ul li ul li a")
    return {a.get_text(strip=True): urljoin(BASE_URL, a["href"]) for a in links}


def parse_book(article, category):
    """Pull the five raw fields out of one <article class="product_pod">."""
    title_tag = article.select_one("h3 a")
    price_tag = article.select_one("p.price_color")
    rating_tag = article.select_one("p.star-rating")
    avail_tag = article.select_one("p.availability")

    # star-rating is stored as a CSS class, e.g. class="star-rating Three"
    rating_classes = rating_tag.get("class", []) if rating_tag else []
    star_rating = next((c for c in rating_classes if c != "star-rating"), None)

    return {
        "title": title_tag.get("title") if title_tag else None,
        "price": price_tag.get_text(strip=True) if price_tag else None,
        "star_rating": star_rating,
        "availability": avail_tag.get_text(strip=True) if avail_tag else None,
        "category": category,
    }


def scrape_category(session, name, url):
    """Walk every page of one category, following the 'next' link."""
    books = []
    while url:
        soup = get_soup(session, url)
        books.extend(parse_book(a, name) for a in soup.select("article.product_pod"))
        next_link = soup.select_one("li.next a")
        url = urljoin(url, next_link["href"]) if next_link else None
        time.sleep(0.3)  # be polite to the practice site
    return books


def main():
    session = requests.Session()
    category_urls = find_category_urls(session)
    rows = []
    for name in CATEGORIES:
        if name not in category_urls:
            print(f"  ! category '{name}' not found on site, skipping")
            continue
        books = scrape_category(session, name, category_urls[name])
        print(f"  {name:<20} {len(books)} books")
        rows.extend(books)

    df = pd.DataFrame(rows)
    RAW_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    print(f"Saved {len(df)} raw rows across {df['category'].nunique()} categories -> {RAW_CSV}")


if __name__ == "__main__":
    main()

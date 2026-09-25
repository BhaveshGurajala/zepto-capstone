"""Step 5 - SQL queries run against books.db.

Each entry is (title, clauses shown, SQL). Together they cover
SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN, BETWEEN and JOIN.
"""

QUERIES = [
    (
        "Q1 - Five-star books (SELECT / WHERE)",
        "SELECT, WHERE",
        """
SELECT title, price_gbp, rating
FROM books
WHERE rating = 5;
""",
    ),
    (
        "Q2 - 10 most expensive books (ORDER BY / LIMIT)",
        "SELECT, ORDER BY, LIMIT",
        """
SELECT title, price_gbp, price_inr
FROM books
ORDER BY price_gbp DESC
LIMIT 10;
""",
    ),
    (
        "Q3 - Rating values that actually occur (DISTINCT)",
        "SELECT DISTINCT, ORDER BY",
        """
SELECT DISTINCT rating
FROM books
ORDER BY rating;
""",
    ),
    (
        "Q4 - Books priced between GBP 20 and GBP 30 (BETWEEN)",
        "SELECT, WHERE, BETWEEN, ORDER BY",
        """
SELECT title, price_gbp, price_inr
FROM books
WHERE price_gbp BETWEEN 20 AND 30
ORDER BY price_gbp;
""",
    ),
    (
        "Q5 - Books rated 4 or 5 stars (IN)",
        "SELECT, WHERE, IN, ORDER BY, LIMIT",
        """
SELECT title, rating, price_gbp
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, price_gbp ASC
LIMIT 15;
""",
    ),
    (
        "Q6 - Highly rated books with their category name (JOIN)",
        "SELECT, JOIN, WHERE, ORDER BY",
        """
SELECT c.category_name, b.title, b.rating, b.price_gbp, b.price_inr
FROM books AS b
JOIN categories AS c ON b.category_id = c.category_id
WHERE b.rating >= 4
ORDER BY c.category_name, b.rating DESC, b.price_gbp DESC, b.title;
""",
    ),
    (
        "Q7 - Book count and average price per category (JOIN + GROUP BY)",
        "SELECT, JOIN, GROUP BY, ORDER BY",
        """
SELECT c.category_name,
       COUNT(*)                   AS n_books,
       ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp,
       ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
       ROUND(AVG(b.rating), 2)    AS avg_rating
FROM books AS b
JOIN categories AS c ON b.category_id = c.category_id
GROUP BY c.category_name
ORDER BY n_books DESC;
""",
    ),
]

# The JOIN query that step 6 reproduces with pd.merge
JOIN_QUERY_INDEX = 5

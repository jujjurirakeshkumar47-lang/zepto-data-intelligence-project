import requests
import time
import pandas as pd
import sqlite3
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# ==========================================
# 1. SCRAPE 5 PAGES
# ==========================================

base_url = "https://books.toscrape.com/"
all_books = []

for page_number in range(1, 6):

    if page_number == 1:
        page_url = base_url
    else:
        page_url = urljoin(
            base_url,
            f"catalogue/page-{page_number}.html"
        )

    response = requests.get(
        page_url,
        timeout=20
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    books = soup.find_all(
        "article",
        class_="product_pod"
    )

    for book in books:

        title = book.h3.a["title"]

        price = book.find(
            "p",
            class_="price_color"
        ).get_text(strip=True)

        star_rating = book.find(
            "p",
            class_="star-rating"
        )["class"][1]

        availability = book.select_one(
            "p.instock.availability"
        ).get_text(" ", strip=True)

        # Get book detail page
        book_url = urljoin(
            page_url,
            book.h3.a["href"]
        )

        book_response = requests.get(
            book_url,
            timeout=20
        )

        book_response.raise_for_status()

        book_soup = BeautifulSoup(
            book_response.text,
            "html.parser"
        )

        # Get category from breadcrumb
        category_links = book_soup.select(
            "ul.breadcrumb li a"
        )

        if category_links:
            category = category_links[-1].get_text(
                strip=True
            )
        else:
            category = None

        all_books.append({
            "title": title,
            "price": price,
            "star_rating": star_rating,
            "availability": availability,
            "category": category
        })

        # Small delay between requests
        time.sleep(0.2)


print("Total books:", len(all_books))


# ==========================================
# 2. CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(all_books)

print("\nRaw Data:")
print(df.head())


# ==========================================
# 3. CLEAN PRICE
# ==========================================

df["price_gbp"] = (
    df["price"]
    .str.replace("Â£", "", regex=False)
    .str.replace("£", "", regex=False)
)

# Convert invalid values to NaN instead of crashing
df["price_gbp"] = pd.to_numeric(
    df["price_gbp"],
    errors="coerce"
)


# ==========================================
# 4. CLEAN STAR RATING
# ==========================================

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["star_rating"].map(
    rating_map
)


# ==========================================
# 5. CLEAN AVAILABILITY
# ==========================================

df["in_stock"] = df[
    "availability"
].str.contains(
    "In stock",
    case=False,
    na=False
)


# ==========================================
# 6. GBP TO INR
# ==========================================

# Fixed conversion rate required by assignment
# 1 GBP = 105.50 INR

df["price_inr"] = (
    df["price_gbp"] * 105.50
)


# ==========================================
# 7. REMOVE INVALID REQUIRED DATA
# ==========================================

before_cleaning = len(df)

df = df.dropna(
    subset=[
        "title",
        "price_gbp",
        "rating",
        "category"
    ]
)

after_cleaning = len(df)

print(
    "\nRows removed during cleaning:",
    before_cleaning - after_cleaning
)


# ==========================================
# 8. DISPLAY CLEANED DATA
# ==========================================

print("\nCleaned Data:")

print(
    df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ].head()
)

print("\nShape:")
print(df.shape)


# ==========================================
# 9. CREATE SQLITE DATABASE
# ==========================================

connection = sqlite3.connect(
    "books.db"
)

cursor = connection.cursor()

cursor.execute(
    "PRAGMA foreign_keys = ON"
)


# ==========================================
# 10. RESET DATABASE
# ==========================================

# Drop old tables so rerunning the pipeline
# does not create duplicate records.

cursor.execute(
    "DROP TABLE IF EXISTS books"
)

cursor.execute(
    "DROP TABLE IF EXISTS categories"
)


# ==========================================
# 11. CREATE CATEGORIES TABLE
# ==========================================

cursor.execute("""
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
)
""")


# ==========================================
# 12. CREATE BOOKS TABLE
# ==========================================

cursor.execute("""
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")


# ==========================================
# 13. INSERT CATEGORIES
# ==========================================

for category in df["category"].unique():

    cursor.execute(
        """
        INSERT OR IGNORE INTO categories
        (category_name)
        VALUES (?)
        """,
        (category,)
    )


# ==========================================
# 14. INSERT BOOKS
# ==========================================

for _, row in df.iterrows():

    cursor.execute(
        """
        SELECT category_id
        FROM categories
        WHERE category_name = ?
        """,
        (row["category"],)
    )

    category_result = cursor.fetchone()

    if category_result is None:
        continue

    category_id = category_result[0]

    cursor.execute(
        """
        INSERT INTO books
        (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            int(row["in_stock"]),
            category_id
        )
    )


connection.commit()


# ==========================================
# 15. CHECK DATABASE
# ==========================================

cursor.execute(
    "SELECT COUNT(*) FROM books"
)

book_count = cursor.fetchone()[0]

cursor.execute(
    "SELECT COUNT(*) FROM categories"
)

category_count = cursor.fetchone()[0]


print("\nDatabase created successfully!")

print(
    "Books in database:",
    book_count
)

print(
    "Categories in database:",
    category_count
)


# ==========================================
# 16. SQL QUERY 1
# SELECT + WHERE
# ==========================================

query1 = """
SELECT title, price_gbp
FROM books
WHERE in_stock = 1;
"""

result1 = pd.read_sql(
    query1,
    connection
)

print("\nQuery 1 - In-stock books:")
print(result1.head(10))


# ==========================================
# 17. SQL QUERY 2
# ORDER BY + LIMIT
# ==========================================

query2 = """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10;
"""

result2 = pd.read_sql(
    query2,
    connection
)

print("\nQuery 2 - Most expensive books:")
print(result2)


# ==========================================
# 18. SQL QUERY 3
# DISTINCT
# ==========================================

query3 = """
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;
"""

result3 = pd.read_sql(
    query3,
    connection
)

print("\nQuery 3 - Categories:")
print(result3)


# ==========================================
# 19. SQL QUERY 4
# BETWEEN
# ==========================================

query4 = """
SELECT title, price_gbp
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;
"""

result4 = pd.read_sql(
    query4,
    connection
)

print(
    "\nQuery 4 - Books between £20 and £40:"
)

print(result4.head(10))


# ==========================================
# 20. SQL QUERY 5
# JOIN
# ==========================================

query5 = """
SELECT
    b.title,
    b.price_gbp,
    b.rating,
    c.category_name
FROM books b
JOIN categories c
ON b.category_id = c.category_id
ORDER BY b.rating DESC, b.title ASC
LIMIT 10;
"""

result5 = pd.read_sql(
    query5,
    connection
)

print("\nQuery 5 - JOIN:")
print(result5)


# ==========================================
# 21. REPRODUCE JOIN USING PD.MERGE
# ==========================================

books_df = pd.read_sql(
    """
    SELECT
        book_id,
        title,
        price_gbp,
        rating,
        category_id
    FROM books
    """,
    connection
)

categories_df = pd.read_sql(
    """
    SELECT
        category_id,
        category_name
    FROM categories
    """,
    connection
)


# Perform Pandas MERGE

merge_result = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)


# Keep same columns as SQL JOIN

merge_result = merge_result[
    [
        "title",
        "price_gbp",
        "rating",
        "category_name"
    ]
]


# Use same sorting as SQL

merge_result = merge_result.sort_values(
    by=[
        "rating",
        "title"
    ],
    ascending=[
        False,
        True
    ]
).head(10)


# Reset index

merge_result = merge_result.reset_index(
    drop=True
)


# ==========================================
# 22. COMPARE SQL JOIN AND PANDAS MERGE
# ==========================================

sql_join_result = result5[
    [
        "title",
        "price_gbp",
        "rating",
        "category_name"
    ]
].reset_index(
    drop=True
)


print("\nSQL JOIN Result:")
print(sql_join_result)


print("\nPandas MERGE Result:")
print(merge_result)


match = sql_join_result.equals(
    merge_result
)


print(
    "\nSQL JOIN and Pandas MERGE match:",
    match
)


# ==========================================
# 23. SAVE SQL QUERY STRINGS AND OUTPUTS
# ==========================================

os.makedirs(
    "query_outputs",
    exist_ok=True
)


with open(
    "query_outputs/query_results.txt",
    "w",
    encoding="utf-8"
) as file:

    queries_and_results = [
        ("QUERY 1 - SELECT + WHERE", query1, result1),
        ("QUERY 2 - ORDER BY + LIMIT", query2, result2),
        ("QUERY 3 - DISTINCT", query3, result3),
        ("QUERY 4 - BETWEEN", query4, result4),
        ("QUERY 5 - JOIN", query5, result5)
    ]

    for query_name, query, result in queries_and_results:

        file.write(
            "\n====================================\n"
        )

        file.write(
            query_name + "\n"
        )

        file.write(
            "====================================\n\n"
        )

        file.write(
            "SQL QUERY:\n"
        )

        file.write(
            query
        )

        file.write(
            "\n\nOUTPUT:\n"
        )

        file.write(
            result.to_string(index=False)
        )

        file.write(
            "\n\n"
        )


# ==========================================
# 24. CLOSE DATABASE
# ==========================================

connection.close()


# ==========================================
# 25. FINAL MESSAGE
# ==========================================

print(
    "\nQuery outputs saved to:"
)

print(
    "query_outputs/query_results.txt"
)

print(
    "\nPipeline completed successfully!"
)
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import numpy as np
import sqlite3
import time


# website
base_url = "https://books.toscrape.com/"

# converting rating words to numbers
rating_numbers = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

books_data = []


# scrape first 5 pages
for page in range(1, 6):

    page_url = urljoin(base_url, f"catalogue/page-{page}.html")

    response = requests.get(page_url)
    soup = BeautifulSoup(response.text, "html.parser")

    books = soup.find_all("article", class_="product_pod")

    for book in books:

        # get title and book page link
        link = book.find("h3").find("a")

        title = link["title"]
        book_url = urljoin(page_url, link["href"])

        # get price
        price = book.find("p", class_="price_color").text

        # get star rating
        rating = book.find("p", class_="star-rating")["class"][1]

        # get stock information
        availability = book.find(
            "p", class_="instock availability"
        ).text.strip()

        # go to the book page to get category
        book_response = requests.get(book_url)
        book_soup = BeautifulSoup(book_response.text, "html.parser")

        breadcrumb = book_soup.find(
            "ul", class_="breadcrumb"
        ).find_all("li")

        category = breadcrumb[2].text.strip()

        books_data.append({
            "title": title,
            "price_text": price,
            "rating_word": rating,
            "availability_text": availability,
            "category": category
        })

        # small delay between requests
        time.sleep(0.1)

    print(
        f"Page {page} done. "
        f"{len(books)} books found. "
        f"Total: {len(books_data)}"
    )


print("\nTotal books scraped:", len(books_data))


# -------------------------
# PUT DATA INTO DATAFRAME
# -------------------------

df = pd.DataFrame(books_data)

print("Shape:", df.shape)
print(df.head())


# -------------------------
# CLEAN PRICE
# -------------------------

# remove pound symbol
df["price_gbp"] = df["price_text"].str.replace(
    "£", "",
    regex=False
)

# change price to number
df["price_gbp"] = pd.to_numeric(
    df["price_gbp"],
    errors="coerce"
)

# count missing prices
bad_prices = df["price_gbp"].isna().sum()

# fill missing prices using median
df["price_gbp"] = df["price_gbp"].fillna(
    df["price_gbp"].median()
)

print("Bad prices filled with median:", bad_prices)


# -------------------------
# CLEAN RATING
# -------------------------

df["rating"] = df["rating_word"].map(rating_numbers)

before = len(df)

# remove rows where rating did not work
df = df.dropna(subset=["rating"])

df["rating"] = df["rating"].astype(int)

after = len(df)

print("Rows removed because of rating:", before - after)
print("Rows left:", after)


# -------------------------
# STOCK COLUMN
# -------------------------

df["in_stock"] = df["availability_text"].str.contains(
    "In stock",
    case=False,
    na=False
)

print(df["in_stock"].value_counts())


# -------------------------
# CONVERT GBP TO INR
# -------------------------

gbp_to_inr = 105.50

df["price_inr"] = (
    df["price_gbp"] * gbp_to_inr
).round(2)

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


# -------------------------
# CREATE SQLITE DATABASE
# -------------------------

conn = sqlite3.connect("books.db")
cursor = conn.cursor()


# delete old tables if they already exist
cursor.execute("DROP TABLE IF EXISTS books")
cursor.execute("DROP TABLE IF EXISTS categories")


# categories table
cursor.execute("""
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE
)
""")


# books table
cursor.execute("""
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id)
    REFERENCES categories(category_id)
)
""")

conn.commit()

print("Tables created")


# -------------------------
# LOAD CATEGORIES
# -------------------------

unique_categories = sorted(df["category"].unique())

categories_df = pd.DataFrame({
    "category_name": unique_categories
})

categories_df.to_sql(
    "categories",
    conn,
    if_exists="append",
    index=False
)


# read categories back from database
category_lookup = pd.read_sql(
    "SELECT * FROM categories",
    conn
)


# join category id onto main dataframe
df = df.merge(
    category_lookup,
    left_on="category",
    right_on="category_name",
    how="left"
)


# -------------------------
# LOAD BOOKS
# -------------------------

books_to_insert = df[
    [
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category_id"
    ]
].copy()


# SQLite uses 1 and 0 instead of True and False
books_to_insert["in_stock"] = (
    books_to_insert["in_stock"].astype(int)
)


books_to_insert.to_sql(
    "books",
    conn,
    if_exists="append",
    index=False
)


print("Categories loaded:", len(category_lookup))


book_count = pd.read_sql(
    "SELECT COUNT(*) AS total FROM books",
    conn
)

print(
    "Books loaded:",
    book_count.iloc[0, 0]
)


# -------------------------
# Q1 - SELECT AND WHERE
# books that are in stock
# -------------------------

query_1 = """
SELECT title, price_gbp
FROM books
WHERE in_stock = 1
LIMIT 10;
"""

result_1 = pd.read_sql(query_1, conn)

print("\nQ1")
print(result_1)


# -------------------------
# Q2 - ORDER BY AND LIMIT
# top 10 highest rated books
# -------------------------

query_2 = """
SELECT title, rating
FROM books
ORDER BY rating DESC, book_id ASC
LIMIT 10;
"""

result_2 = pd.read_sql(query_2, conn)

print("\nQ2")
print(result_2)


# -------------------------
# Q3 - DISTINCT
# all unique categories
# -------------------------

query_3 = """
SELECT DISTINCT category_name
FROM categories;
"""

result_3 = pd.read_sql(query_3, conn)

print("\nQ3")
print(result_3)


# -------------------------
# Q4 - BETWEEN
# books between £20 and £40
# -------------------------

query_4 = """
SELECT title, price_gbp
FROM books
WHERE price_gbp BETWEEN 20 AND 40;
"""

result_4 = pd.read_sql(query_4, conn)

print("\nQ4")
print(result_4)


# -------------------------
# Q5 - JOIN
# highest rated books with category
# -------------------------

query_5 = """
SELECT b.title,
       b.rating,
       c.category_name
FROM books b
JOIN categories c
ON b.category_id = c.category_id
ORDER BY b.rating DESC, b.book_id ASC
LIMIT 10;
"""

result_5_sql = pd.read_sql(query_5, conn)

print("\nQ5 using SQL")
print(result_5_sql)


# -------------------------
# DO Q5 AGAIN USING PANDAS
# -------------------------

books_full = pd.read_sql(
    "SELECT * FROM books",
    conn
)


merged = books_full.merge(
    category_lookup,
    on="category_id",
    how="left"
)


merged = merged.sort_values(
    ["rating", "book_id"],
    ascending=[False, True]
).head(10)


result_5_pandas = merged[
    [
        "title",
        "rating",
        "category_name"
    ]
].reset_index(drop=True)


print("\nQ5 using pandas")
print(result_5_pandas)


# compare SQL result and pandas result
same_result = (
    result_5_sql.reset_index(drop=True)
    .equals(result_5_pandas)
)

print(
    "\nDo SQL and pandas give the same answer?",
    same_result
)


# close database connection
conn.close()
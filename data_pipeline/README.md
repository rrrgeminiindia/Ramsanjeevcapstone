# Data Pipeline

This is my Module 1 data pipeline project using books.toscrape.com.

I scraped book data from the website, cleaned it and stored it in SQLite database.

## Files

- book_pipeline.py
- books.db
- query_outputs.txt
- README.md
- requirements.txt

## Data Source

I used:

https://books.toscrape.com/

I scraped first 5 catalogue pages using requests and BeautifulSoup.

Around 100 books are collected.

## Data Collected

For every book I collected:

- title
- price
- rating
- availability
- category

## Data Cleaning

For price I removed the £ symbol and converted it to float.

If any price could not be converted I used median price.

Ratings like:

One, Two, Three, Four, Five

were converted into numbers 1 to 5.

If rating was invalid that row was removed.

For availability, books with "In stock" were converted to True and other values to False.

## Currency Conversion

I used fixed conversion rate given in the question:

1 GBP = 105.50 INR

INR price was calculated using:

price_inr = price_gbp * 105.50

No currency API was used.

## Database

I used SQLite database called:

books.db

I created 2 tables.

categories:

- category_id
- category_name

books:

- book_id
- title
- price_gbp
- price_inr
- rating
- in_stock
- category_id

category_id in books is a foreign key connected to categories table.

## SQL Queries

I created 5 SQL queries.

They include:

- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- BETWEEN
- JOIN

The query results are also saved in:

query_outputs.txt

## Pandas

I used pd.read_sql() to read SQL query results into pandas.

I also recreated the SQL JOIN using pd.merge().

Then I compared SQL result and pandas result to check if both are same.

## How to run

I used VS Code for this project.

First activate the virtual env.

```powershell
.\data_pipeline\.venv\Scripts\Activate.ps1

pip install requests beautifulsoup4 pandas numpy

run python book_pipeline.py
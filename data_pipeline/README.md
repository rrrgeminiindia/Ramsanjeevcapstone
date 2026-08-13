# Data Pipeline

## Overview

This module scrapes book data from books.toscrape.com, cleans the
scraped data, converts book prices from GBP to INR, stores the data
in a normalized SQLite database, and analyzes the database using
SQL and pandas.

## Data Source

https://books.toscrape.com/

The first five catalogue pages are scraped automatically using
requests and BeautifulSoup.

## Data Collected

For each book the following information is collected:

- Title
- Price
- Star rating
- Availability
- Category

At least 60 books are required. This implementation scrapes
approximately 100 books from the first five catalogue pages.

## Data Cleaning

Price:
The £ symbol is removed and the value is converted to float.
If a price cannot be parsed, the missing value is replaced with
the median book price.

Rating:
Text ratings One, Two, Three, Four, and Five are converted to
integer values 1 through 5.
Rows with invalid ratings are removed.

Availability:
Availability text containing "In stock" is converted to True.
Other values are converted to False.

## Currency Conversion

The project-defined fixed conversion rate is:

1 GBP = 105.50 INR

price_inr is calculated using:

price_inr = price_gbp * 105.50

No external currency API is required.

## Database Design

The SQLite database contains two normalized tables:

categories

- category_id - Primary Key
- category_name - Unique category name

books

- book_id - Primary Key
- title
- price_gbp
- price_inr
- rating
- in_stock
- category_id - Foreign Key referencing categories

## SQL Queries

Five SQL queries demonstrate:

- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- BETWEEN
- JOIN

Query results are displayed in the notebook.

## Pandas and SQL Comparison

SQL query results are loaded using pd.read_sql().

The JOIN query is also reproduced using pandas pd.merge().

The SQL and pandas results are compared to verify that they
produce equivalent output.

## Installation

Install the required libraries:

pip install requests beautifulsoup4 pandas numpy

SQLite support is included with Python.

## Running

Open data_pipeline.ipynb in Jupyter Notebook or Google Colab.

Run all cells from top to bottom.

The notebook will:

1. Scrape the website.
2. Clean the data.
3. Convert GBP prices to INR.
4. Create books.db.
5. Create categories and books tables.
6. Insert the cleaned data.
7. Execute SQL queries.
8. Compare SQL JOIN results with pandas merge results.
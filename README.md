Use the main repo `README.md` as a short overview of the complete capstone and point to the three modules.

Replace the empty root `README.md` with this:

````markdown
# Ramsanjeev Capstone Project

This repository contains my capstone project with three modules covering data engineering, data analytics, machine learning, and a RAG-based support assistant.

## Project Structure

```text
Ramsanjeevcapstone/
│
├── data_pipeline/
│   ├── book_pipeline.py
│   ├── books.db
│   ├── query_outputs.txt
│   ├── requirements.txt
│   └── README.md
│
├── analytics/
│   ├── analytics_pipeline.py
│   ├── titanic.csv
│   ├── requirements.txt
│   └── README.md
│
├── support_assistant/
│   ├── docs/
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   ├── doc_03.txt
│   │   ├── doc_04.txt
│   │   ├── doc_05.txt
│   │   ├── doc_06.txt
│   │   ├── doc_07.txt
│   │   └── doc_08.txt
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
└── README.md
````

## Module 1 - Data Pipeline

The first module builds a complete data pipeline using Python.

Books are scraped from Books to Scrape and cleaned using Python and Pandas.

The project performs:

* Web scraping
* Data cleaning
* GBP to INR price conversion
* SQLite database creation
* Normalized tables
* SQL queries
* Pandas analysis
* SQL JOIN and Pandas merge

The complete implementation and instructions are available inside:

`data_pipeline/README.md`

## Module 2 - Titanic Analytics

The second module performs exploratory data analysis and machine learning using the Titanic dataset.

The project includes:

* Missing value analysis
* Outlier detection
* Univariate and multivariate analysis
* Data visualization
* Feature preprocessing
* Logistic Regression
* Decision Tree
* Random Forest
* Class imbalance handling
* SMOTE
* GridSearchCV
* ROC-AUC comparison
* Linear Regression
* Model evaluation
* Saved machine learning pipeline

The complete analysis and results are available inside:

`analytics/README.md`

## Module 3 - Zepto Support Assistant

The third module implements a RAG-based customer support assistant for Zepto policies.

The project uses:

* Sentence Transformers
* `all-MiniLM-L6-v2`
* ChromaDB
* LangGraph
* FastAPI
* Pydantic
* Docker

Eight Zepto policy documents are embedded and stored inside ChromaDB.

LangGraph classifies the user question into either a policy question or a general question.

For policy questions, the system retrieves the top three relevant documents from ChromaDB and generates a deterministic response in mock mode.

The FastAPI endpoint is:

```text
POST /ask
```

The application runs in mock mode by default:

```text
MOCK_LLM=1
```

No external LLM API key is required.

The complete implementation and API examples are available inside:

`support_assistant/README.md`

## Technologies Used

Python
Pandas
BeautifulSoup
SQLite
Matplotlib
Seaborn
Scikit-learn
Imbalanced-learn
Sentence Transformers
ChromaDB
LangGraph
FastAPI
Pydantic
Docker
Git and GitHub

## Repository

All three capstone modules are maintained in this single GitHub repository.

Each module contains its own README with detailed implementation, execution instructions, and results.

````

One small correction before committing: make sure the Q2 filename really is `analytics_pipeline.py` and that `analytics/README.md` has that exact capitalization. If your actual filenames differ, use the actual names in the root README.

Then save it and run:

```powershell
git add README.md
git commit -m "Update main project README"
git push origin main
````

That is enough for the **main repository README**.

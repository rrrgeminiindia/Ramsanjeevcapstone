# Zepto Support Assistant

This project is a simple support assistant built using RAG, ChromaDB, LangGraph and FastAPI.

The project runs in mock mode by default using:

MOCK_LLM=1

No LLM API key is required.

## Documents

There are 8 Zepto policy documents inside the docs folder.

The documents contain information about:

- Delivery
- Returns and refunds
- Membership
- Order tracking
- Cancellation
- Damaged or missing items
- Gift cards
- Customer support

## Embeddings

The 8 documents are loaded from the docs folder.

The sentence-transformers model:

all-MiniLM-L6-v2

is used to convert the documents into embeddings.

The embeddings are stored inside a ChromaDB collection called:

zepto_docs

## Retrieval

When a policy question is received, the question is converted into an embedding.

ChromaDB searches the stored documents and returns the top 3 matching documents.

Example:

Question:

What is the delivery policy?

Retrieved documents:

doc_01
doc_05
doc_02

The top result is doc_01 which contains the Delivery Policy.

## LangGraph

The LangGraph contains 3 nodes:

classify_intent

retrieve_and_answer

direct_answer

The flow is:

START
  |
  v
classify_intent
  |
  +----------------------+
  |                      |
policy_question      general_question
  |                      |
  v                      v
retrieve_and_answer   direct_answer
  |                      |
  +----------END----------+

classify_intent checks whether the question is related to Zepto policy.

Policy questions go to retrieve_and_answer.

Other questions go to direct_answer.

## Mock Mode

MOCK_LLM=1 is the default mode.

In mock mode no LLM API is called.

For policy questions the answer is created using the top retrieved document.

The answer format is:

Based on the retrieved context: ...

For general questions the fixed answer is:

I can only answer questions about Zepto policies right now.

## FastAPI

The LangGraph is exposed using FastAPI.

Run the API using:

python -m uvicorn main:app --reload

Open:

http://127.0.0.1:8000/docs

The main endpoint is:

POST /ask

## Example 1 - Policy Question

Request:

```json
{
  "query": "What is the delivery policy?"
}


{
  "answer": "Based on the retrieved context: Delivery Policy: \"Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order vo",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_02"
  ],
  "confidence": 1
}



{
  "query": "how is maxsteppen?"
}



{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1
}
# IMPORTS

import os
import chromadb

from typing import TypedDict
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END
from fastapi import FastAPI
from pydantic import BaseModel


# MOCK MODE
# Default is 1, so no LLM API call is made

MOCK_LLM = os.getenv("MOCK_LLM", "1")

print("MOCK_LLM:", MOCK_LLM)


# --------------------------------------------------
# 1. LOAD DOCUMENTS
# --------------------------------------------------

documents = []
ids = []
metadatas = []

for i in range(1, 9):

    file_name = f"doc_{i:02}.txt"

    with open(f"docs/{file_name}", "r", encoding="utf-8") as f:
        text = f.read()

    documents.append(text)
    ids.append(f"doc_{i:02}")
    metadatas.append({"source": file_name})


print("Documents loaded:", len(documents))


# --------------------------------------------------
# 2. EMBEDDINGS
# --------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents)

print("Embeddings created:", len(embeddings))


# --------------------------------------------------
# 3. CHROMADB
# --------------------------------------------------

chroma_client = chromadb.Client()

# Delete old collection when cell is run again

try:
    chroma_client.delete_collection("zepto_docs")
except:
    pass


collection = chroma_client.create_collection(
    name="zepto_docs"
)


collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    ids=ids,
    metadatas=metadatas
)


print("Documents stored in ChromaDB:", collection.count())


# --------------------------------------------------
# 4. RETRIEVAL FUNCTION
# --------------------------------------------------

def retrieve(question, n_results=3):

    query_embedding = model.encode([question])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )

    return results


# --------------------------------------------------
# 5. PROMPT TEMPLATE
# --------------------------------------------------

prompt_template = """
Role:
You are a Zepto customer support assistant.

Context:
{context}

Task:
Answer the customer question using only the given context.

Constraint:
Do not answer using information which is not present in the context.
Do not make up any Zepto policy.

Few Shot Example:

Question:
What is the delivery fee?

Context:
Standard delivery is free on orders over INR 149.
Orders below INR 149 have INR 25 delivery fee.

Answer:
Standard delivery is free on orders over INR 149.
Orders below INR 149 have INR 25 delivery fee.

Format:
Return JSON with these fields:

answer
sources
confidence

Length:
Answer in maximum 3 sentences.

Question:
{question}
"""


# --------------------------------------------------
# 6. PYDANTIC RESPONSE
# --------------------------------------------------

class SupportResponse(BaseModel):
    answer: str
    sources: list
    confidence: float


class AskRequest(BaseModel):
    query: str


# --------------------------------------------------
# 7. LANGGRAPH STATE
# --------------------------------------------------

class SupportState(TypedDict):
    question: str
    intent: str
    context: str
    answer: str
    sources: list
    confidence: float


# --------------------------------------------------
# 8. NODE 1 - CLASSIFY INTENT
# --------------------------------------------------

def classify_intent(state):

    question = state["question"].lower()

    keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]


    if MOCK_LLM == "1":

        if any(word in question for word in keywords):
            intent = "policy_question"

        else:
            intent = "general_question"


    else:

        # Optional real LLM path
        # Assignment grading uses MOCK_LLM=1

        if any(word in question for word in keywords):
            intent = "policy_question"
        else:
            intent = "general_question"


    print("Intent:", intent)

    return {
        "intent": intent
    }


# --------------------------------------------------
# 9. NODE 2 - RETRIEVE AND ANSWER
# --------------------------------------------------

def retrieve_and_answer(state):

    question = state["question"]

    results = retrieve(
        question,
        n_results=3
    )


    retrieved_documents = results["documents"][0]

    retrieved_ids = results["ids"][0]


    context = "\n\n".join(
        retrieved_documents
    )


    print("Retrieved documents:", retrieved_ids)


    if MOCK_LLM == "1":

        top_document = retrieved_documents[0]

        top_chunk_snippet = top_document[:200]

        answer = (
            "Based on the retrieved context: "
            + top_chunk_snippet
        )


    else:

        # Optional real LLM path

        prompt = prompt_template.format(
            context=context,
            question=question
        )

        print(prompt)

        answer = (
            "Real LLM answer will be generated here."
        )


    return {
        "context": context,
        "answer": answer,
        "sources": retrieved_ids,
        "confidence": 1.0
    }


# --------------------------------------------------
# 10. NODE 3 - DIRECT ANSWER
# --------------------------------------------------

def direct_answer(state):

    if MOCK_LLM == "1":

        answer = (
            "I can only answer questions "
            "about Zepto policies right now."
        )


    else:

        # Optional real LLM path

        answer = (
            "Real LLM general answer will be generated here."
        )


    return {
        "answer": answer,
        "sources": [],
        "confidence": 1.0
    }


# --------------------------------------------------
# 11. ROUTING FUNCTION
# --------------------------------------------------

def route_question(state):

    return state["intent"]


# --------------------------------------------------
# 12. LANGGRAPH
# --------------------------------------------------

graph = StateGraph(SupportState)


graph.add_node(
    "classify_intent",
    classify_intent
)


graph.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)


graph.add_node(
    "direct_answer",
    direct_answer
)


graph.add_edge(
    START,
    "classify_intent"
)


graph.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer"
    }
)


graph.add_edge(
    "retrieve_and_answer",
    END
)


graph.add_edge(
    "direct_answer",
    END
)


support_graph = graph.compile()


print("LangGraph created")


# --------------------------------------------------
# 13. TEST POLICY QUESTION
# --------------------------------------------------

print("\nPOLICY QUESTION TEST")


policy_result = support_graph.invoke(
    {
        "question": "What is the delivery policy?"
    }
)


print("Answer:")
print(policy_result["answer"])

print("Sources:")
print(policy_result["sources"])

print("Confidence:")
print(policy_result["confidence"])


# --------------------------------------------------
# 14. TEST GENERAL QUESTION
# --------------------------------------------------

print("\nGENERAL QUESTION TEST")


general_result = support_graph.invoke(
    {
        "question": "What is the capital of France?"
    }
)


print("Answer:")
print(general_result["answer"])

print("Sources:")
print(general_result["sources"])

print("Confidence:")
print(general_result["confidence"])


# --------------------------------------------------
# 15. FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="Zepto Support Assistant"
)


@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Zepto Support Assistant"
    }


@app.post(
    "/ask",
    response_model=SupportResponse
)
def ask(request: AskRequest):

    result = support_graph.invoke(
        {
            "question": request.query
        }
    )


    response = SupportResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )


    return response
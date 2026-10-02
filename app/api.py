from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from .ingest import Document
from .pipeline import RAGPipeline

app = FastAPI(title="Document Intelligence RAG API", version="0.1.0")

DEMO_DOCS = [
    Document(
        source="policy.txt",
        text=(
            "Refunds are available within 30 days of purchase when a receipt is provided. "
            "Digital services that have already been fully consumed are not refundable."
        ),
    ),
    Document(
        source="shipping.txt",
        text=(
            "Standard shipping takes three to five business days. "
            "Express shipping takes one to two business days."
        ),
    ),
]
PIPELINE = RAGPipeline.from_documents(DEMO_DOCS)

class AskRequest(BaseModel):
    question: str
    top_k: int = 4

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/ask")
def ask(request: AskRequest) -> dict:
    answer = PIPELINE.ask(request.question, request.top_k)
    return {
        "answer": answer.text,
        "citations": answer.citations,
        "context": answer.context,
    }

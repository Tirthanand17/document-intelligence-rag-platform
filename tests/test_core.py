from app.chunking import chunk_text
from app.evaluation import RetrievalCase, hit_rate
from app.ingest import Document
from app.pipeline import RAGPipeline

def test_chunking_creates_overlap_chunks():
    chunks = chunk_text("demo.txt", "A" * 1500, chunk_size=500, overlap=100)
    assert len(chunks) >= 3
    assert all(c.source == "demo.txt" for c in chunks)

def test_rag_retrieves_correct_source():
    docs = [
        Document("refunds.txt", "Refunds are available for 30 days with a receipt."),
        Document("shipping.txt", "Express shipping takes one to two business days."),
    ]
    rag = RAGPipeline.from_documents(docs)
    answer = rag.ask("How long does express shipping take?", top_k=1)
    assert answer.citations == ["shipping.txt"]

def test_hit_rate():
    cases = [
        RetrievalCase("refund?", "refunds.txt"),
        RetrievalCase("shipping?", "shipping.txt"),
    ]
    results = [["refunds.txt"], ["shipping.txt", "refunds.txt"]]
    assert hit_rate(cases, results) == 1.0

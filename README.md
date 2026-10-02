# Production Document Intelligence RAG Platform

Portfolio-grade AI/ML project combining Data Science, NLP, retrieval-augmented generation (RAG), deep-learning embeddings, OCR/PDF ingestion, evaluation, API delivery, testing, and automation.

## Why this project

Many AI jobs need more than a notebook model. They need an end-to-end system that can ingest company documents, retrieve evidence, answer with citations, expose an API, and be evaluated and deployed reliably.

## Architecture

PDF / TXT / Markdown
-> Document Loader
-> NLP Chunking
-> Embedding Backend (TF-IDF baseline or transformer embeddings)
-> Vector Retrieval
-> Cited Answer Layer
-> Evaluation
-> FastAPI
-> Docker / CI

## AI/ML capabilities

- NLP document chunking
- semantic retrieval architecture
- optional deep-learning embeddings using sentence-transformers/all-MiniLM-L6-v2
- RAG architecture
- cited answer output
- retrieval hit-rate evaluation
- FastAPI inference endpoint
- Docker deployment
- automated pytest coverage
- GitHub Actions CI

## Quick start

~~~bash
python -m pip install -r requirements.txt
uvicorn app.api:app --reload
~~~

## Optional transformer embeddings

The default test path uses a deterministic TF-IDF baseline so CI stays light and reproducible.

~~~bash
python -m pip install -r requirements-transformers.txt
~~~

Use SentenceTransformerEmbedder from app.embeddings when building the vector index.

## OCR / scanned PDFs

The ingestion layer reads searchable PDFs directly. For scanned PDFs, this project is designed to sit behind an OCR preprocessing step such as OCRmyPDF before indexing.

Full pipeline:

Scanned PDF -> OCR -> text extraction -> chunking -> embeddings -> retrieval -> cited answer

## Tests

~~~bash
pytest -q
~~~

## Portfolio relevance

This project demonstrates work relevant to:
- AI/ML engineering
- Data Science
- NLP
- Deep Learning inference
- LLM/RAG applications
- AI automation
- document intelligence
- Python backend/API development
- model/retrieval evaluation
- MLOps-ready deployment patterns

## Next production extensions

- persistent vector database (FAISS / Chroma / pgvector)
- configurable LLM provider
- reranking
- metadata filters
- groundedness evaluation
- OCR preprocessing service
- authentication and multi-tenant collections
- observability and latency/cost dashboards

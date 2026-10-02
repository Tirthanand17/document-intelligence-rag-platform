from __future__ import annotations
from dataclasses import dataclass
from .chunking import chunk_text
from .generation import Answer, extractive_answer
from .ingest import Document
from .retrieval import VectorIndex

@dataclass
class RAGPipeline:
    index: VectorIndex

    @classmethod
    def from_documents(cls, documents: list[Document]) -> "RAGPipeline":
        chunks = []
        for doc in documents:
            chunks.extend(chunk_text(doc.source, doc.text))
        index = VectorIndex()
        index.build(chunks)
        return cls(index=index)

    def ask(self, question: str, top_k: int = 4) -> Answer:
        return extractive_answer(question, self.index.search(question, top_k=top_k))

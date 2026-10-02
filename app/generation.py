from __future__ import annotations
from dataclasses import dataclass
from .retrieval import SearchHit

@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[str]
    context: list[str]

def extractive_answer(question: str, hits: list[SearchHit]) -> Answer:
    if not hits:
        return Answer(
            text="I could not find relevant evidence in the indexed documents.",
            citations=[],
            context=[],
        )
    top = hits[:3]
    context = [h.chunk.text for h in top]
    citations = [h.chunk.source for h in top]
    answer = "Based on the indexed documents, the most relevant evidence is: " + " ".join(context)
    return Answer(text=answer, citations=citations, context=context)

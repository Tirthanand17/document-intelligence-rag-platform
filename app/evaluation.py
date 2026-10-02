from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RetrievalCase:
    query: str
    expected_source: str

def hit_rate(cases: list[RetrievalCase], results: list[list[str]]) -> float:
    if not cases:
        return 0.0
    hits = 0
    for case, sources in zip(cases, results):
        hits += int(case.expected_source in sources)
    return hits / len(cases)

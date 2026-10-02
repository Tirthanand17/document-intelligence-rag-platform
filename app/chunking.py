from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str

def chunk_text(source: str, text: str, chunk_size: int = 700, overlap: int = 120) -> list[Chunk]:
    clean = " ".join(text.split())
    if not clean:
        return []
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        piece = clean[start:end].strip()
        if piece:
            chunks.append(Chunk(chunk_id=f"{source}:{index}", source=source, text=piece))
            index += 1
        if end == len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks

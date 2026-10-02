from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from pypdf import PdfReader

@dataclass(frozen=True)
class Document:
    source: str
    text: str

def load_text(path: Path) -> Document:
    return Document(source=path.name, text=path.read_text(encoding="utf-8"))

def load_pdf(path: Path) -> Document:
    reader = PdfReader(str(path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return Document(source=path.name, text=text)

def load_document(path: str | Path) -> Document:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    if path.suffix.lower() in {".txt", ".md"}:
        return load_text(path)
    raise ValueError(f"Unsupported document type: {path.suffix}")

def load_many(paths: Iterable[str | Path]) -> list[Document]:
    return [load_document(p) for p in paths]

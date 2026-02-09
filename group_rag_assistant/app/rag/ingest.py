from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma

from app.config import SETTINGS
from app.llm import get_embeddings

TEXT_EXT = {".txt", ".md", ".tex", ".bib", ".csv"}


def load_docs(kb_dir: Path):
    docs = []
    for p in kb_dir.rglob("*"):
        if p.is_dir():
            continue
        ext = p.suffix.lower()
        if ext == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())  # page metadata should exist
        elif ext in TEXT_EXT:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
    return docs


def build_index() -> None:
    SETTINGS.kb_dir.mkdir(parents=True, exist_ok=True)
    docs = load_docs(SETTINGS.kb_dir)
    if not docs:
        print("kb/ is empty. Add at least one .md/.txt/.pdf first.")
        return

    kb_root = SETTINGS.kb_dir.resolve()

    for d in docs:
        d.metadata = d.metadata or {}

        # Ensure a consistent "source" key
        src = (
            d.metadata.get("source")
            or d.metadata.get("file_path")
            or d.metadata.get("path")
            or d.metadata.get("filename")
        )
        if src is not None:
            d.metadata["source"] = str(src)

        # Robust kb_scope: public/internal based on first folder under kb/
        scope = "unknown"
        try:
            rel = Path(d.metadata.get("source", "")).resolve().relative_to(kb_root)
            top = rel.parts[0] if rel.parts else ""
            if top.lower() == "internal":
                scope = "internal"
            elif top.lower() == "public":
                scope = "public"
        except Exception:
            pass
        d.metadata["kb_scope"] = scope

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    SETTINGS.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(SETTINGS.vectorstore_dir),
    )
    print(f"OK: indexed {len(chunks)} chunks into {SETTINGS.vectorstore_dir}")


if __name__ == "__main__":
    build_index()

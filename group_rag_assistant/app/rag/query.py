from __future__ import annotations

from typing import List

from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import SETTINGS
from app.llm import get_chat, get_embeddings


def format_sources(docs: List, max_snippet: int = 220) -> str:
    lines = []
    seen = set()

    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)

        snippet = (d.page_content or "").replace("\n", " ").strip()
        if len(snippet) > max_snippet:
            snippet = snippet[:max_snippet] + "…"

        if page is not None:
            key = f"{src}:{int(page)}"
            line = f"- {src} (p.{int(page)+1}): {snippet}"
        else:
            key = f"{src}"
            line = f"- {src}: {snippet}"

        if key in seen:
            continue
        seen.add(key)
        lines.append(line)

    return "\n".join(lines)


SYSTEM_HINT = {
    "thesis": (
        "Use ONLY the provided context. If context is insufficient, output [MISSING INFO] "
        "and list exactly what is missing. Do not invent citations or data."
    ),
    "reviewer": (
        "Use ONLY the provided context/text. Be strict and specific. "
        "Do not invent facts or citations."
    ),
    "python": (
        "Return runnable Python code only. Use the provided context when relevant. "
        "If you must assume something, write it as a Python comment starting with: # ASSUMPTION:"
    ),
}


def ask(agent: str, question: str, k: int = 8) -> dict:
    vs = Chroma(
        persist_directory=str(SETTINGS.vectorstore_dir),
        embedding_function=get_embeddings(),
    )

    # Better retrieval: MMR for diversity
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": max(24, k * 3),
            "lambda_mult": 0.6,
        },
    )

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    sys = SYSTEM_HINT.get(agent, SYSTEM_HINT["thesis"])
    prompt = (
        "Context:\n"
        f"{context}\n\n"
        "User request:\n"
        f"{question}\n"
    )

    llm = get_chat(agent)
    resp = llm.invoke([
        SystemMessage(content=sys),
        HumanMessage(content=prompt),
    ])

    return {
        "answer": resp.content,
        "sources": format_sources(docs),
    }

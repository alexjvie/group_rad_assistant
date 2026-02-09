# group_rag_assistant
# Group RAG Assistant (Local)

Local-only assistant with three modes: Writer (internal: `thesis`), Python (internal: `python`), Reviewer (internal: `reviewer`). Workflow: (1) Put documents into `kb/` (2) Build the index into `vectorstore/` via `python main.py ingest` (3) Use the Web UI or CLI to ask questions (RAG). The Web UI is intentionally read-only (no upload/reindex) to protect long-term datasets and avoid accidental changes.

Prerequisites: Install Ollama (required). Verify with: `ollama --version`.

Setup (fresh clone):
1) `git clone <REPO_URL>`
2) `cd group_rag_assistant`
3) `python3 -m venv .venv`
4) `source .venv/bin/activate`
5) `pip install -U pip`
6) `pip install -r requirements.txt`

Download base models (Ollama):
- `ollama pull llama3.2:3b`
- `ollama pull qwen2.5-coder:3b`
- `ollama pull nomic-embed-text`

Create the custom “typewriter” models (recommended):
- `ollama create typewriter-python -f modelfiles/TypewriterPython.Modelfile`
- `ollama create typewriter-thesis -f modelfiles/TypewriterWriter.Modelfile`
- `ollama create typewriter-reviewer -f modelfiles/TypewriterReviewer.Modelfile`
Check with: `ollama list`.

Add documents to the knowledge base (`kb/`): put PDF/MD/TXT/TEX/BIB/CSV files into `kb/` (e.g., `kb/papers/`, `kb/notes/`).

Build/update the vector index (`vectorstore/`): run `python main.py ingest`. Re-run this whenever you change files in `kb/`.

Run the Web UI:
- Start: `python -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload`
- Open: http://127.0.0.1:8000
If the UI says “Vector index not found”, run `python main.py ingest`.

Run via CLI (no browser):
- Writer: `python main.py thesis "Write a short Methods paragraph based on the KB."`
- Python: `python main.py python "Write code that loads a CSV (cycle_index, capacity_ah) and plots capacity vs cycle."`
- Reviewer: `python main.py reviewer "Review this paragraph for overclaims and missing citations: <paste text>"`

Saving outputs (optional): `mkdir -p reports` then save CLI outputs directly, e.g. `python main.py thesis "Write a Methods paragraph ..." > reports/methods.md` and `python main.py python "Write Python code for plotting ..." > reports/plot_script.py`. From the Web UI: generate output → click Copy → paste into a file under `reports/` in PyCharm.

from pathlib import Path
from app.config import SETTINGS
from app.rag.ingest import build_index
from app.rag.query import ask

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

def ensure_index():
    if not SETTINGS.vectorstore_dir.exists() or not any(SETTINGS.vectorstore_dir.iterdir()):
        build_index()

def run_demo():
    ensure_index()

    # 1) Thesis agent: schreibt kurzen Methods-Absatz aus KB
    q1 = "Write a short Methods-style paragraph for an electrochemistry data analysis workflow. Use the retrieved context. No invented citations."
    out1 = ask("thesis", q1)
    (REPORT_DIR / "methods.md").write_text(out1["answer"].strip() + "\n", encoding="utf-8")

    # 2) Python agent: liefert Analyseplan + plotting steps (ohne echte Daten)
    q2 = (
        "Propose a robust Python data analysis skeleton for battery cycling data (capacity vs cycle). "
        "Include: loading, cleaning, feature extraction, and matplotlib plotting. "
        "Return code only."
    )
    out2 = ask("python", q2)
    (REPORT_DIR / "analysis_skeleton.py").write_text(out2["answer"].strip() + "\n", encoding="utf-8")

    # 3) Optional: Reviewer (hier als “thesis” missbraucht, bis du ein extra reviewer-Modell baust)
    q3 = (
        "Review the following Methods paragraph for scientific tone, missing info, and potential hallucinations. "
        "Return bullet points only.\n\n"
        + out1["answer"]
    )
    out3 = ask("reviewer", q3)
    (REPORT_DIR / "review.txt").write_text(out3["answer"].strip() + "\n", encoding="utf-8")

    print("OK: wrote reports/methods.md, reports/analysis_skeleton.py, reports/review.txt")

if __name__ == "__main__":
    run_demo()

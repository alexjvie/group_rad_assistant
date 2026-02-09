import sys

from app.config import SETTINGS
from app.rag.ingest import build_index
from app.rag.query import ask
from app.workflows.demo_agent import run_demo


USAGE = """
python main.py ingest
python main.py thesis "question"
python main.py python "question"
python main.py demo
"""


def ensure_index() -> None:
    if not SETTINGS.vectorstore_dir.exists() or not any(SETTINGS.vectorstore_dir.iterdir()):
        build_index()


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        return

    cmd = sys.argv[1].lower().strip()

    if cmd == "ingest":
        build_index()
        return

    if cmd == "demo":
        # demo_agent already calls ensure_index internally, but no harm either way
        run_demo()
        return

    if cmd in {"thesis", "python", "reviewer"}:
        q = " ".join(sys.argv[2:]).strip()
        if not q:
            print("Missing question.\n", USAGE)
            return

        ensure_index()
        out = ask(cmd, q)
        print(out["answer"])
        if out.get("sources"):
            print("\nSOURCES:\n", out["sources"])
        return

    print(USAGE)


if __name__ == "__main__":
    main()

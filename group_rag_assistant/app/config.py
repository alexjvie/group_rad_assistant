from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    kb_dir: Path = ROOT / "kb"
    vectorstore_dir: Path = ROOT / "vectorstore"

    embed_model: str = "nomic-embed-text"
    thesis_model: str = "typewriter-thesis"
    python_model: str = "typewriter-python"
    reviewer_model: str = "typewriter-thesis"  # optional (kannst du später ersetzen)

SETTINGS = Settings()

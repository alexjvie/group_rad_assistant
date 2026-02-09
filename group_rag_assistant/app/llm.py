from langchain_ollama import ChatOllama, OllamaEmbeddings
from app.config import SETTINGS

def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=SETTINGS.embed_model)

def get_chat(agent: str) -> ChatOllama:
    if agent == "thesis":
        return ChatOllama(model=SETTINGS.thesis_model, temperature=0.25)
    if agent == "python":
        return ChatOllama(model=SETTINGS.python_model, temperature=0.2)
    if agent == "reviewer":
        return ChatOllama(model=SETTINGS.reviewer_model, temperature=0.2)
    raise ValueError(f"Unknown agent: {agent}")

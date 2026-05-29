import pytest

from app.config import settings


def test_vectorstore_path_exists():
    assert settings.vectorstore_path.exists(), (
        f"Vectorstore not found at {settings.vectorstore_path}. "
        "Run 'python -m app.ingest' first."
    )


def test_get_retriever():
    from app.retriever import get_retriever
    retriever = get_retriever()
    assert retriever is not None
    docs = retriever.invoke("test query")
    assert isinstance(docs, list)

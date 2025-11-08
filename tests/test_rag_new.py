# tests/test_rag_new.py
import os
import pytest
from src.agent.rag.retriever import build_retriever

need_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY no configurada"
)

@need_key
def test_retriever_loads():
    retriever = build_retriever(
        manuals_dir="./data/manuales",
        index_dir="./data/rag_index",
        k=2,
        embedding_model="text-embedding-3-small"
    )
    results = retriever.invoke("test")
    assert isinstance(results, list)

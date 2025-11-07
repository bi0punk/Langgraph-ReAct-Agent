# tests/test_rag.py
# Pruebas de los loaders (sin red) y del retriever (se omite si no hay API Key)
import os
from pathlib import Path
import pytest

from src.agent.rag.loaders import load_documents

# Solo importamos retriever si hay API (evitamos fallas en CI sin credenciales)
HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if HAS_OPENAI:
    from src.agent.rag.retriever import build_retriever


def test_load_documents_tmpdir(tmp_path: Path):
    # Crea archivos temporales de prueba
    man = tmp_path / "manuales"
    man.mkdir()
    (man / "a.txt").write_text("hola txt", encoding="utf-8")
    (man / "b.md").write_text("# titulo\nhola md", encoding="utf-8")
    # No probamos PDF real aquí para mantener la prueba liviana

    docs = load_documents(str(man))
    contents = [d.page_content for d in docs]
    assert any("hola txt" in c for c in contents)
    assert any("hola md" in c for c in contents)


@pytest.mark.skipif(not HAS_OPENAI, reason="Requiere OPENAI_API_KEY para embeddings/FAISS")
def test_build_retriever_end_to_end(tmp_path: Path):
    # Pequeño índice con 2 docs
    man = tmp_path / "manuales"
    idx = tmp_path / "rag_index"
    man.mkdir()
    (man / "c.txt").write_text("este es un manual de prueba sobre cables azules", encoding="utf-8")
    (man / "d.txt").write_text("otro manual que habla de conectores rojos", encoding="utf-8")

    ret = build_retriever(str(man), str(idx), k=2)
    results = ret.invoke("cables azules")
    assert len(results) >= 1
    assert "cables azules" in results[0].page_content

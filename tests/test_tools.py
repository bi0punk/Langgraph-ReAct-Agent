# tests/test_tools.py
import re
from datetime import datetime

from src.agent.tools import calculator, now_time, manual_search, set_retriever


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, q: str):
        # Ignora la query y devuelve docs fijos para la prueba
        return self._docs


class _Doc:
    """Documento mínimo compatible con manual_search (tiene .page_content y .metadata)."""
    def __init__(self, content, source="fake.txt", page=None):
        self.page_content = content
        self.metadata = {"source": source}
        if page is not None:
            self.metadata["page"] = page


def test_calculator_basic():
    assert calculator.run("2+2") == "4"
    assert calculator.run("sqrt(81)") == "9.0"


def test_now_time_format():
    ts = now_time.run("")
    # Debe parecer un ISO8601 simple: YYYY-MM-DDTHH:MM:SS
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", ts)


def test_manual_search_with_fake_retriever():
    docs = [
        _Doc("Contenido A", source="man/testA.md", page=1),
        _Doc("Contenido B", source="man/testB.md", page=2),
    ]
    set_retriever(_FakeRetriever(docs))
    out = manual_search.run("cualquier consulta")
    assert "Contenido A" in out and "Contenido B" in out
    assert "man/testA.md [p.1]" in out
    assert "man/testB.md [p.2]" in out

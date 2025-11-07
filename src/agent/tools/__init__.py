# src/agent/tools/__init__.py
# ------------------------------------------------------------
# Exportamos las tools para que app.py pueda importarlas fácil.
# ------------------------------------------------------------
from .calculator import calculator
from .now_time import now_time
from .manual_search import manual_search, set_retriever

__all__ = [
    "calculator",
    "now_time",
    "manual_search",
    "set_retriever",
]

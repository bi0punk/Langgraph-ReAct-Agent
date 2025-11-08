from .calculator import calculator
from .now_time import now_time
from .manual_search import manual_search, set_retriever
from .system_stats import system_stats   # <-- NUEVO

__all__ = [
    "calculator",
    "now_time",
    "manual_search",
    "set_retriever",
    "system_stats",                      # <-- NUEVO
]

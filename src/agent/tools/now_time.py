# src/agent/tools/now_time.py
# ------------------------------------------------------------
# Tool: now_time
# Retorna la hora/fecha del sistema en formato ISO.
# ------------------------------------------------------------

from langchain_core.tools import tool
from datetime import datetime


@tool("now_time")
def now_time(_: str = "") -> str:
    """
    Retorna la fecha/hora actual ISO8601.
    No requiere argumentos.
    """
    return datetime.now().isoformat(timespec="seconds")

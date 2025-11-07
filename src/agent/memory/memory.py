# src/agent/memory/memory.py
# -------------------------------------------------------------------
# Fábrica de "checkpointers" para LangGraph.
#
# Por defecto usamos MemorySaver (en memoria del proceso).
# Opcionalmente, si está disponible la dependencia, se puede usar SqliteSaver
# para persistir el estado entre reinicios del proceso.
#
# Uso típico en app.py:
#   from .memory.memory import make_checkpointer, thread_config
#   checkpointer = make_checkpointer()  # lee env/config
#   app = graph.compile(checkpointer=checkpointer)
#   ...
#   app.invoke(state, config=thread_config("usuario-123"))
# -------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Optional, Dict, Any

from langgraph.checkpoint.memory import MemorySaver

# SqliteSaver es opcional; solo lo activamos si existe el módulo
try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
    _HAS_SQLITE = True
except Exception:
    SqliteSaver = None  # type: ignore
    _HAS_SQLITE = False


def make_checkpointer(
    *,
    kind: Optional[str] = None,
    sqlite_path: Optional[str] = None,
):
    """
    Crea y retorna un "checkpointer" para LangGraph.

    Args:
        kind:         "memory" (default) o "sqlite".
                      Si no se especifica, se toma de la variable de entorno CHECKPOINTER.
        sqlite_path:  ruta del archivo .sqlite para el modo "sqlite".
                      Si no se especifica, se toma de la variable CHECKPOINT_DB
                      (default: "./data/checkpoints.sqlite").

    Env:
        CHECKPOINTER   → "memory" | "sqlite"  (por defecto: "memory")
        CHECKPOINT_DB  → ruta del archivo sqlite (por defecto: "./data/checkpoints.sqlite")

    Returns:
        Instancia de MemorySaver o SqliteSaver.

    Notas:
        - "memory": rápido y sencillo, pero NO persiste entre reinicios del proceso.
        - "sqlite": persiste estados y es útil para entornos de desarrollo/QA.
                    Requiere que 'langgraph.checkpoint.sqlite' esté disponible en tu versión.
    """
    kind = (kind or os.getenv("CHECKPOINTER", "memory")).strip().lower()

    if kind == "sqlite":
        if not _HAS_SQLITE:
            raise RuntimeError(
                "CHECKPOINTER=sqlite pero 'SqliteSaver' no está disponible en esta versión "
                "de LangGraph. Actualiza langgraph o usa CHECKPOINTER=memory."
            )
        sqlite_path = sqlite_path or os.getenv("CHECKPOINT_DB", "./data/checkpoints.sqlite")
        # Aseguramos directorio
        os.makedirs(os.path.dirname(os.path.abspath(sqlite_path)), exist_ok=True)
        return SqliteSaver(sqlite_path)  # type: ignore

    # Fallback/por defecto
    return MemorySaver()


def thread_config(thread_id: str, *, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Helper para construir el 'config' que se pasa a app.invoke(...) con el thread_id
    y metadatos opcionales. Mantener este formato centralizado ayuda a no duplicar lógica.

    Ejemplo:
        app.invoke(
            {"messages": [...]},
            config=thread_config("usuario-123", metadata={"channel": "cli"})
        )

    Args:
        thread_id: identificador lógico de la conversación/sesión.
        metadata:  diccionario opcional con metadatos adicionales (no obligatorio).

    Returns:
        dict con 'configurable' y 'metadata' (si se proporcionó).
    """
    cfg: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if metadata:
        cfg["metadata"] = metadata
    return cfg


__all__ = ["make_checkpointer", "thread_config"]

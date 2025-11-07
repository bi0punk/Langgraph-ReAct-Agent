# src/agent/state.py
# ------------------------------------------------------------
from typing import TypedDict, List, Annotated, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages  # <-- acumulador correcto


class AgentState(TypedDict):
    """
    Estado que se pasa entre nodos del grafo.

    Campos:
      - messages: historial conversacional (acumulado con add_messages)
      - steps:    contador de iteraciones del ciclo act→observe→reason
      - rag_ctx:  contexto opcional inyectado por RAG
      - meta:     metadatos libres
    """
    messages: Annotated[List[BaseMessage], add_messages]  # <-- clave del fix
    steps: int
    rag_ctx: Optional[str]
    meta: Optional[Dict[str, Any]]


DEFAULT_MAX_STEPS: int = 6


def new_state(
    *,
    messages: Optional[List[BaseMessage]] = None,
    steps: int = 0,
    rag_ctx: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> AgentState:
    return AgentState(
        messages=messages or [],
        steps=steps,
        rag_ctx=rag_ctx,
        meta=meta or {},
    )

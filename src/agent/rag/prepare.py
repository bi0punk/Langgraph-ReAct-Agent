# src/agent/rag/prepare.py
# ------------------------------------------------------------
# Nodo opcional 'rag_prepare' que:
#   - Toma el último mensaje del usuario
#   - Recupera pasajes con el retriever
#   - Aplica umbral de relevancia (threshold) para decidir si usar RAG
#   - Si relevante → inyecta contexto; si no → no interfiere
# ------------------------------------------------------------

from __future__ import annotations
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from ..state import AgentState


def _format_rag_context(docs) -> Optional[str]:
    """
    Decide si el contexto recuperado es relevante.
    - Si no hay documentos → no usar RAG
    - Si el score del top-1 es bajo → ignorar RAG
    - Caso contrario → concatenar contenidos
    """
    if not docs:
        return None

    top = docs[0]
    score = top.metadata.get("score", 0.0)

    # Ajusta según necesidad (entre 0.15 y 0.35 normalmente va bien)
    if score < 0.25:
        return None

    return "\n\n".join(d.page_content for d in docs)


def rag_prepare_node_factory(retriever):
    """
    Retorna el nodo rag_prepare(state).
    Se inserta antes de 'act' solo si graph.rag_mode = 'auto' o 'both'.
    """
    def rag_prepare(state: AgentState):
        # Obtener último mensaje humano
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        last_user = user_msgs[-1].content if user_msgs else ""
        docs = retriever.invoke(last_user) if last_user else []

        ctx = _format_rag_context(docs)

        # Si no hay contexto relevante → no inyectamos nada
        if not ctx:
            return {"rag_ctx": None}

        # Inyectar contexto como mensaje del sistema (leído por el prompt)
        sys_msg = SystemMessage(content=f"RAG_CONTEXT:\n\n{ctx}")

        return {
            "messages": [sys_msg],   # se acumula gracias a add_messages en AgentState
            "rag_ctx": ctx,
        }

    return rag_prepare

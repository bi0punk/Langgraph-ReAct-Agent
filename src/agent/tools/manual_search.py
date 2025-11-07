# src/agent/tools/manual_search.py
# ------------------------------------------------------------
# Tool: manual_search
# Llama al retriever para obtener pasajes relevantes.
#
# Nota importante:
#   El retriever se "inyecta" desde app.py mediante set_retriever().
#   Esto nos permite que esta tool no dependa directamente del RAG.
# ------------------------------------------------------------

from typing import Optional
from langchain_core.tools import tool

# Este retriever será configurado en app.py
_RETRIEVER = None


def set_retriever(retriever):
    """
    Debe llamarse desde app.py para conectar el retriever real.
    """
    global _RETRIEVER
    _RETRIEVER = retriever


@tool("manual_search")
def manual_search(query: str) -> str:
    """
    Busca pasajes relevantes en los manuales.
    Si no hay retriever configurado, devuelve aviso.
    """
    if _RETRIEVER is None:
        return "⚠️ manual_search no tiene retriever configurado aún."

    if not query.strip():
        return "⚠️ Debes ingresar una consulta no vacía."

    docs = _RETRIEVER.invoke(query)
    if not docs:
        return "No se encontraron pasajes relevantes en los manuales."

    out = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "desconocido")
        page = d.metadata.get("page", None)
        loc = f"{src}" + (f" [p.{page}]" if page is not None else "")
        out.append(f"[{i}] {loc}\n{d.page_content}")

    return "\n\n".join(out)

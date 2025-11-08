# src/agent/app.py
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from langchain.agents import create_agent as create_react_agent


from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.llm.factory import make_chat_llm
from src.agent.prompts.templates import get_prompt
from src.agent.tools import (
    calculator,
    now_time,
    manual_search,
    set_retriever,
    system_stats,
)
from src.agent.rag.retriever import build_retriever
from src.agent.memory.memory import make_checkpointer  # memory | sqlite

log = logging.getLogger("agent.app")


def _deep_get(d: Dict[str, Any], path: str, default: Any) -> Any:
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_settings() -> Dict[str, Any]:
    # Acepta settings.yaml|yml|setting.yaml|yml y aplica defaults si falta
    candidates = [
        Path("./config/settings.yaml"),
        Path("./config/settings.yml"),
        Path("./config/setting.yaml"),
        Path("./config/setting.yml"),
    ]
    cfg: Dict[str, Any] = {}
    for p in candidates:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            break

    # Defaults mínimos
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("manuals_dir", "./data/manuales")
    cfg["paths"].setdefault("index_dir", "./data/rag_index")

    cfg.setdefault("llm", {})
    cfg["llm"].setdefault("chat_model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    cfg["llm"].setdefault("temperature", float(os.getenv("OPENAI_TEMPERATURE", "0.0")))
    cfg["llm"].setdefault("request_timeout", int(os.getenv("OPENAI_TIMEOUT", "60")))

    cfg.setdefault("embeddings", {})
    cfg["embeddings"].setdefault("model", os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"))

    cfg.setdefault("rag", {})
    cfg["rag"].setdefault("top_k", int(os.getenv("RAG_TOP_K", "4")))
    cfg["rag"].setdefault("threshold", float(os.getenv("RAG_THRESHOLD", "0.25")))

    cfg.setdefault("graph", {})
    cfg["graph"].setdefault("rag_mode", os.getenv("RAG_MODE", "both"))  # none|auto|both

    cfg.setdefault("memory", {})
    cfg["memory"].setdefault("checkpointer", os.getenv("CHECKPOINTER", "memory"))

    return cfg


def _format_rag_context(docs, threshold: float) -> Optional[str]:
    """
    Devuelve contexto si el top-1 supera el umbral de relevancia; si no, None.
    """
    if not docs:
        return None
    top = docs[0]
    score = top.metadata.get("score", 0.0)
    if score < threshold:
        return None
    return "\n\n".join(d.page_content for d in docs)


def get_app():
    settings = load_settings()

    manuals_dir = _deep_get(settings, "paths.manuals_dir", "./data/manuales")
    index_dir   = _deep_get(settings, "paths.index_dir", "./data/rag_index")
    rag_k       = int(_deep_get(settings, "rag.top_k", 4))
    rag_mode    = str(_deep_get(settings, "graph.rag_mode", "both")).lower()
    threshold   = float(_deep_get(settings, "rag.threshold", 0.25))
    checkpointer_kind = str(_deep_get(settings, "memory.checkpointer", "memory"))

    # LLM
    llm = make_chat_llm(
        model=str(_deep_get(settings, "llm.chat_model", "gpt-4o-mini")),
        temperature=float(_deep_get(settings, "llm.temperature", 0.0)),
        timeout=int(_deep_get(settings, "llm.request_timeout", 60)),
    )

    # Prompt (tu system + MessagesPlaceholder)
    prompt = get_prompt()

    # RAG (retriever) + tool manual_search
    retriever = build_retriever(
        manuals_dir=manuals_dir,
        index_dir=index_dir,
        k=rag_k,
        embedding_model=str(_deep_get(settings, "embeddings.model", "text-embedding-3-small")),
    )
    set_retriever(retriever)

    # Tools disponibles
    tools = [calculator, now_time, manual_search, system_stats]

    # ---- pre_model_hook para inyectar RAG contexto ANTES del modelo (si aplica) ----
    # En la API que muestra tu firma, pre_model_hook puede ser un callable que
    # recibe el "input state" y devuelve:
    #   - un dict con claves a mezclar (por ej. {"messages": [...]})
    #   - o el input modificado
    #
    # Aquí devolvemos solo {"messages": [SystemMessage(...)]} cuando corresponde.
    def pre_hook(state, _runtime=None):
        if rag_mode not in ("auto", "both"):
            return {}
        msgs = state.get("messages", [])
        if not msgs:
            return {}
        # último mensaje humano
        q = ""
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                q = m.content
                break
        if not q:
            return {}
        docs = retriever.invoke(q)
        ctx = _format_rag_context(docs, threshold)
        if not ctx:
            return {}
        return {"messages": [SystemMessage(content=f"RAG_CONTEXT:\n\n{ctx}")]}

    # Memoria / checkpointing
    checkpointer = make_checkpointer(kind=checkpointer_kind)

    # >>> Prebuilt ReAct agent (compilado directamente) <<<
    app = create_react_agent(
        model=llm,                 # <- tu firma exige 'model'
        tools=tools,
        prompt=prompt,             # tu system prompt + placeholder
        pre_model_hook=pre_hook,   # inyección RAG (auto/both) con threshold
        checkpointer=checkpointer, # conserva memoria por thread_id
        # puedes pasar state_schema/context_schema si lo necesitas
    )

    log.info(
        "✅ Agente (prebuilt ReAct) listo. RAG mode=%s (thr=%.2f), memory=%s",
        rag_mode, threshold, checkpointer_kind
    )
    return app

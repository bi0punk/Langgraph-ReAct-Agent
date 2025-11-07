# src/agent/app.py
# -------------------------------------------------------------------
from __future__ import annotations

import os
import logging
import yaml
from pathlib import Path

from langgraph.prebuilt import ToolNode
from src.agent.state import AgentState
from src.agent.memory.memory import make_checkpointer
from src.agent.llm.factory import make_chat_llm
from src.agent.prompts.templates import get_prompt
from src.agent.tools import calculator, now_time, manual_search, set_retriever
from src.agent.rag.retriever import build_retriever
from src.agent.rag.prepare import rag_prepare_node_factory
from src.agent.graph.act_observe_reason import build_act_observe_reason_graph

log = logging.getLogger("agent.app")


def load_settings():
    cfg_path = Path("./config/settings.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError("❌ Falta config/settings.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_app():
    settings = load_settings()

    manuals_dir = settings["paths"]["manuals_dir"]
    index_dir   = settings["paths"]["index_dir"]
    rag_k       = settings["rag"]["top_k"]
    rag_mode    = settings["graph"]["rag_mode"]
    max_steps   = settings["graph"]["max_steps"]
    checkpointer_kind = settings["memory"]["checkpointer"]

    # ---- LLM principal ----
    llm = make_chat_llm(
        model=settings["llm"]["chat_model"],
        temperature=float(settings["llm"]["temperature"]),
        timeout=int(settings["llm"]["request_timeout"]),
    )
    llm_act = llm.bind_tools([calculator, now_time, manual_search])  # para ACT
    llm_reason = llm                                               # para REASON

    # ---- Prompt ----
    prompt = get_prompt()

    # ---- RAG ----
    retriever = build_retriever(manuals_dir, index_dir, k=rag_k)
    set_retriever(retriever)

    tool_node = ToolNode([calculator, now_time, manual_search])

    graph = build_act_observe_reason_graph(
        prompt=prompt,
        llm_act=llm_act,
        llm_reason=llm_reason,
        tool_node=tool_node,
        max_steps=max_steps,
    )

    # ---- RAG automático antes de ACT ----
    if rag_mode.lower() in ("auto", "both"):
        rag_prepare = rag_prepare_node_factory(retriever)
        graph.add_node("rag_prepare", rag_prepare)
        graph.set_entry_point("rag_prepare")
        graph.add_edge("rag_prepare", "act")
    else:
        graph.set_entry_point("act")

    # ---- Memoria ----
    checkpointer = make_checkpointer(kind=checkpointer_kind)

    app = graph.compile(checkpointer=checkpointer)

    log.info("✅ Agente listo.")
    return app


__all__ = ["get_app"]

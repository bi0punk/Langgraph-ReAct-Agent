import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False

from app.config import settings
from app.llm import get_llm
from app.models import AgentState
from app.nodes import (
    generate_node,
    generate_with_tools_node,
    grade_node,
    react_node,
    retrieve_node,
    rewrite_node,
)
from app.tools import retrieve_documents

logger = logging.getLogger(__name__)

tools = [retrieve_documents]
tool_node = ToolNode(tools)
_llm_with_tools = None


def get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = get_llm().bind_tools(tools)
    return _llm_with_tools


def grade_router(state: AgentState) -> str:
    docs = state.get("documents", [])
    if not docs:
        logger.info("No documents retrieved — skipping grade, going to rewrite")
        return "rewrite"
    if state.get("grade") == "relevant":
        return "generate"
    if state.get("rewrite_count", 0) >= settings.max_rewrites:
        logger.info("Max rewrites (%d) reached, forcing generate", settings.max_rewrites)
        return "generate"
    return "rewrite"


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "continue"
    return "end"


def create_pipeline():
    rag_mode = settings.rag_mode
    logger.info("Building pipeline with RAG_MODE=%s", rag_mode)

    if rag_mode == "none":
        return _build_react_agent()
    if rag_mode == "both":
        return _build_both_pipeline()
    return _build_rag_pipeline()


def _build_react_agent():
    def agent(state: AgentState):
        llm = get_llm().bind_tools(tools)
        return react_node(state, llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {
        "continue": "tools",
        "end": END,
    })
    workflow.add_edge("tools", "agent")

    memory = _get_checkpointer()
    return workflow.compile(checkpointer=memory)


def _build_rag_pipeline():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", grade_router, {
        "generate": "generate",
        "rewrite": "rewrite",
    })
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    memory = _get_checkpointer()
    return workflow.compile(checkpointer=memory)


def _build_both_pipeline():
    def _generate_with_tools(state, config=None):
        llm = get_llm_with_tools()
        return generate_with_tools_node(state, llm, config)

    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", _generate_with_tools)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", grade_router, {
        "generate": "generate",
        "rewrite": "rewrite",
    })
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_conditional_edges("generate", should_continue, {
        "continue": "tools",
        "end": END,
    })
    workflow.add_edge("tools", "generate")

    memory = _get_checkpointer()
    return workflow.compile(checkpointer=memory)


def _get_checkpointer():
    if settings.checkpointer == "sqlite" and HAS_SQLITE:
        import sqlite3
        settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(settings.checkpoint_db), check_same_thread=False)
        return SqliteSaver(conn)
    if settings.checkpointer == "sqlite" and not HAS_SQLITE:
        logger.warning("SqliteSaver not installed — falling back to MemorySaver")
    return MemorySaver()

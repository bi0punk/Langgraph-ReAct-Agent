# src/agent/graph/act_observe_reason.py
# -------------------------------------------------------------------
# Grafo del agente con el patrón:
#   act     → el LLM decide (puede emitir tool_calls)
#   observe → se ejecutan las tools y se anexan observaciones (ToolMessages)
#   reason  → el LLM razona con las observaciones y decide responder o volver a actuar
#
# Este módulo solo define el GRAFO. No compila ni crea memoria aquí.
# La compilación (con MemorySaver/Redis, etc.) se hace en src/agent/app.py.
#
# Dependencias externas que se inyectan:
#   - prompt: ChatPromptTemplate con SYSTEM + MessagesPlaceholder("messages")
#   - llm_act: modelo de chat con tools “binded” (para fase ACT)
#   - llm_reason: modelo de chat “normal” (para fase REASON)
#   - tool_node: instancia de langgraph.prebuilt.ToolNode con tus tools
#
# Uso esperado (en app.py):
#   from .graph.act_observe_reason import build_act_observe_reason_graph
#   graph = build_act_observe_reason_graph(prompt, llm_act, llm_reason, tool_node, max_steps=6)
#   app = graph.compile(checkpointer=MemorySaver())  # u otro backend de memoria
# -------------------------------------------------------------------

from typing import Literal

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from ..state import AgentState, DEFAULT_MAX_STEPS


def build_act_observe_reason_graph(
    prompt: ChatPromptTemplate,
    llm_act,          # ChatModel con tools:   llm.bind_tools(TOOLS)
    llm_reason,       # ChatModel sin tools:   llm (para sintetizar/razonar)
    tool_node,        # ToolNode(TOOLS)
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> StateGraph:
    """
    Construye y devuelve un StateGraph con el ciclo act → observe → reason.
    No compila ni asigna memoria. Eso se realiza fuera de este módulo.

    Args:
        prompt:       ChatPromptTemplate con SYSTEM + MessagesPlaceholder("messages")
        llm_act:      modelo con herramientas habilitadas (tool-calling) para la fase ACT
        llm_reason:   modelo sin herramientas para la fase REASON
        tool_node:    ToolNode ya configurado con tus tools
        max_steps:    límite de iteraciones para evitar loops infinitos

    Returns:
        StateGraph listo para compilar.
    """

    # -----------------------------
    # NODOS
    # -----------------------------
    def act(state: AgentState):
        """
        Fase ACT:
        - El modelo decide qué hacer en base al historial (y opcionalmente rag_ctx si tu prompt lo usa).
        - Puede emitir tool_calls (pidiendo ejecutar herramientas).
        - Incrementa el contador de pasos.
        """
        convo = prompt.invoke({"messages": state["messages"]})
        ai = llm_act.invoke(convo.to_messages())
        return {
            "messages": [ai],
            "steps": state.get("steps", 0) + 1,
        }

    def observe(state: AgentState):
        """
        Fase OBSERVE:
        - Ejecuta las herramientas pedidas en ACT.
        - ToolNode se encarga de anexar ToolMessages al historial.
        - No devolvemos cambios manuales aquí (ToolNode actualiza el state).
        """
        return {}

    def reason(state: AgentState):
        """
        Fase REASON:
        - El modelo razona con el historial (incluye observaciones de tools).
        - Puede responder directamente o, si decide, volver a llamar herramientas
          (emitiendo nuevos tool_calls, lo que nos retornará a ACT).
        """
        convo = prompt.invoke({"messages": state["messages"]})
        ai = llm_reason.invoke(convo.to_messages())
        return {"messages": [ai]}

    # -----------------------------
    # ROUTERS
    # -----------------------------
    def route_after_act(state: AgentState) -> Literal["observe", "end"]:
        """
        Después de ACT:
          - Si excede max_steps → end
          - Si el último AIMessage trae tool_calls → observe
          - Si no hay tool_calls → end (ya hay respuesta final)
        """
        if state.get("steps", 0) >= max_steps:
            return "end"
        last = state["messages"][-1]
        has_calls = bool(getattr(last, "tool_calls", None))
        return "observe" if has_calls else "end"

    def route_after_reason(state: AgentState) -> Literal["act", "end"]:
        """
        Después de REASON:
          - Si excede max_steps → end
          - Si el último AIMessage trae tool_calls → act (nuevo ciclo)
          - Si no trae tool_calls → end (respuesta final)
        """
        if state.get("steps", 0) >= max_steps:
            return "end"
        last = state["messages"][-1]
        has_calls = bool(getattr(last, "tool_calls", None))
        return "act" if has_calls else "end"

    # -----------------------------
    # ENSAMBLA EL GRAFO
    # -----------------------------
    graph = StateGraph(AgentState)

    graph.add_node("act", act)
    graph.add_node("observe", tool_node)  # ejecuta tools y añade ToolMessages
    graph.add_node("reason", reason)

    graph.set_entry_point("act")

    # ACT → (tools?) OBSERVE : END
    graph.add_conditional_edges("act", route_after_act, {"observe": "observe", "end": END})
    # OBSERVE → REASON
    graph.add_edge("observe", "reason")
    # REASON → (tools?) ACT : END
    graph.add_conditional_edges("reason", route_after_reason, {"act": "act", "end": END})

    return graph


__all__ = ["build_act_observe_reason_graph"]

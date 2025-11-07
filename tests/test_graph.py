# tests/test_graph.py
# Verifica el ciclo act→observe→reason usando un LLM "falso" que emite tool_calls
# y luego razona con la observación de la tool.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph.act_observe_reason import build_act_observe_reason_graph
from src.agent.tools import calculator, now_time
from src.agent.state import AgentState


class FakeLLMAct:
    """Simula un LLM con tool-calling. Si ve '2+2' pide calculator; si ve 'hora', pide now_time."""
    def invoke(self, messages):
        last_user = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user = m.content
                break

        # Decide qué tool llamar
        tool_calls = []
        if "2+2" in last_user:
            tool_calls = [{"name": "calculator", "args": {"expression": "2+2"}, "id": "call-1"}]
            content = "Voy a calcular 2+2 con la herramienta."
        elif "hora" in last_user:
            tool_calls = [{"name": "now_time", "args": {"" : ""}, "id": "call-2"}]
            content = "Consultaré la hora con la herramienta."
        else:
            content = "No necesito herramientas; responderé directo."

        msg = AIMessage(content=content)
        # Inyectamos el atributo tool_calls que espera ToolNode
        setattr(msg, "tool_calls", tool_calls)
        return msg


class FakeLLMReason:
    """Simula el razonamiento final. Si ve ToolMessages de calculator con '4', responde con el resultado."""
    def invoke(self, messages):
        # Busca salida de tools recientes en la conversación
        tool_output_text = ""
        for m in reversed(messages):
            # ToolNode añade ToolMessage (tipo BaseMessage) con .content = salida de la tool
            # Para esta prueba basta con inspeccionar strings
            if hasattr(m, "content"):
                c = str(m.content)
                if c.strip() == "4":            # resultado de calculator 2+2
                    return AIMessage(content="El resultado es 4.")
                if "T" in c and len(c) >= 19:   # heurística súper simple de fecha ISO
                    return AIMessage(content=f"La hora es {c}.")

        # Si no hubo tools, responde algo fijo
        return AIMessage(content="Respuesta directa (sin tools).")


def test_graph_act_observe_reason_calculator():
    # Prompt mínimo (no cargamos system.txt aquí para evitar I/O)
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Agente de prueba."), MessagesPlaceholder("messages")]
    )

    tool_node = ToolNode([calculator, now_time])

    graph = build_act_observe_reason_graph(
        prompt=prompt,
        llm_act=FakeLLMAct(),
        llm_reason=FakeLLMReason(),
        tool_node=tool_node,
        max_steps=4,
    )

    app = graph.compile(checkpointer=MemorySaver())

    # Disparamos una pregunta que requiere la tool 'calculator'
    state: AgentState = {"messages": [HumanMessage(content="¿Cuánto es 2+2?")], "steps": 0, "rag_ctx": None, "meta": {}}
    result = app.invoke(state, config={"configurable": {"thread_id": "test-1"}})

    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert any("El resultado es 4" in m.content for m in ai_msgs)


def test_graph_act_observe_reason_no_tools():
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Agente de prueba."), MessagesPlaceholder("messages")]
    )
    tool_node = ToolNode([calculator, now_time])

    graph = build_act_observe_reason_graph(
        prompt=prompt,
        llm_act=FakeLLMAct(),
        llm_reason=FakeLLMReason(),
        tool_node=tool_node,
        max_steps=2,
    )
    app = graph.compile(checkpointer=MemorySaver())

    state: AgentState = {"messages": [HumanMessage(content="No uses herramientas.")], "steps": 0, "rag_ctx": None, "meta": {}}
    result = app.invoke(state, config={"configurable": {"thread_id": "test-2"}})

    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert any("Respuesta directa" in m.content for m in ai_msgs)

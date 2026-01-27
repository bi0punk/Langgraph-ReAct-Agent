from typing import TypedDict
from langgraph.graph import StateGraph
from app.rag_chain import generate_rag_answer


class AgentState(TypedDict):
    question: str
    answer: str


def rag_node(state: AgentState):
    answer = generate_rag_answer(state["question"])
    return {"answer": answer}


graph = StateGraph(AgentState)
graph.add_node("rag", rag_node)

graph.set_entry_point("rag")
graph.set_finish_point("rag")

agent = graph.compile()

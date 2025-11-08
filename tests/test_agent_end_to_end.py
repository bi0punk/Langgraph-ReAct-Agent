# tests/test_agent_end_to_end.py
import os
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.app import get_app

need_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY no configurada"
)

@need_key
def test_agent_replies():
    app = get_app()
    result = app.invoke(
    {"messages": [HumanMessage(content="Di 'hola'")]},
    config={"configurable": {"thread_id": "test-run"}}
    )

    assert any(isinstance(m, AIMessage) for m in result["messages"])

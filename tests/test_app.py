# tests/test_app.py
import os
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.app import get_app

need_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY no configurada"
)

@need_key
def test_app_simple_invoke():
    app = get_app()
    result = app.invoke(
    {"messages": [HumanMessage(content="Hola")]},
    config={"configurable": {"thread_id": "test-app"}}
    )

    assert any(isinstance(m, AIMessage) for m in result["messages"])

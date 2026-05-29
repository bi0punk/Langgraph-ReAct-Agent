from typing import Annotated, List, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    documents: List[Document]
    generation: str
    rewrite_count: int
    grade: str

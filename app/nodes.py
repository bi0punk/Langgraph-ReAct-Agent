import json
import logging

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig

from app.config import settings
from app.llm import get_llm
from app.models import AgentState
from app.retriever import get_retriever

logger = logging.getLogger(__name__)

GRADE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a grader evaluating whether retrieved documents are relevant "
        "to the user's question.\n\n"
        "Examples:\n"
        "Question: What is the return policy?\n"
        "Documents: [text about 30-day return policy, refund process]\n"
        "Relevant: true\n\n"
        "Question: What is the capital of France?\n"
        "Documents: [text about laptop specifications]\n"
        "Relevant: false\n\n"
        "Respond with ONLY a JSON object: {{\"relevant\": true}} or {{\"relevant\": false}}."
    )),
    ("human", "Question: {question}\n\nRetrieved documents:\n{documents}"),
])

REWRITE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a query rewriter. Given a user question and a failed search result, "
        "rewrite the question to be more specific and searchable. "
        "Respond with only the rewritten question, nothing else."
    )),
    ("human", "Original question: {question}\n\nPrevious search returned irrelevant results.\nRewritten question:"),
])

GENERATE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant. Answer the user's question based on the "
        "provided context and conversation history. If the context doesn't contain "
        "enough information, say so clearly. Be concise and accurate.\n\n"
        "Conversation history:\n{history}\n\n"
        "Context:\n{context}"
    )),
    ("human", "{question}"),
])


def retrieve_node(state: AgentState) -> dict:
    logger.info("Retrieving documents for: %s", state["question"])
    try:
        retriever = get_retriever()
        documents = retriever.invoke(state["question"])
        logger.info("Retrieved %d documents", len(documents))
        return {"documents": documents, "rewrite_count": state.get("rewrite_count", 0)}
    except FileNotFoundError as e:
        logger.warning("Vectorstore not available: %s", e)
        return {"documents": [], "rewrite_count": state.get("rewrite_count", 0)}


def grade_node(state: AgentState) -> dict:
    docs = state.get("documents", [])
    if not docs:
        logger.info("No documents to grade — skipping")
        return {"grade": "not_relevant"}

    llm = get_llm()
    docs_text = "\n\n".join(d.page_content[:500] for d in docs)
    if not docs_text.strip():
        return {"grade": "not_relevant"}

    prompt = GRADE_TEMPLATE.format_messages(
        question=state["question"],
        documents=docs_text,
    )
    try:
        response = llm.invoke(prompt)
        content = response.content.strip().lower()
    except Exception as e:
        logger.error("LLM grading failed: %s", e)
        return {"grade": "not_relevant"}

    try:
        parsed = json.loads(content)
        relevant = parsed.get("relevant", False)
    except json.JSONDecodeError:
        relevant = "true" in content and "false" not in content

    logger.info("Documents relevant: %s (LLM said: %s)", relevant, content)
    return {"grade": "relevant" if relevant else "not_relevant"}


def rewrite_node(state: AgentState) -> dict:
    llm = get_llm()
    prompt = REWRITE_TEMPLATE.format_messages(question=state["question"])
    try:
        response = llm.invoke(prompt)
        new_query = response.content.strip()
    except Exception as e:
        logger.error("LLM rewrite failed: %s", e)
        return {"question": state["question"], "rewrite_count": state.get("rewrite_count", 0) + 1}
    logger.info("Rewrote query: '%s' -> '%s'", state["question"], new_query)
    return {"question": new_query, "rewrite_count": state.get("rewrite_count", 0) + 1}


def generate_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
    llm = get_llm()
    docs = state.get("documents", [])
    context = "\n\n".join(d.page_content for d in docs) if docs else "No relevant documents found."

    messages = state.get("messages", [])
    history = _format_history(messages)

    prompt = GENERATE_TEMPLATE.format_messages(
        history=history,
        context=context,
        question=state["question"],
    )
    try:
        response = llm.invoke(prompt, config=config if config else None)
        logger.info("Generated answer (%d chars)", len(response.content))
        return {"generation": response.content, "messages": [AIMessage(content=response.content)]}
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        return {"generation": "Lo siento, ocurrió un error al generar la respuesta.", "messages": []}


def generate_with_tools_node(state: AgentState, llm, config: RunnableConfig | None = None) -> dict:
    docs = state.get("documents", [])
    context = "\n\n".join(d.page_content for d in docs) if docs else "No relevant documents found."

    messages = list(state.get("messages", []))
    history = _format_history(messages)

    prompt = GENERATE_TEMPLATE.format_messages(
        history=history,
        context=context,
        question=state["question"],
    )

    response = llm.invoke(prompt, config=config if config else None)

    if hasattr(response, "tool_calls") and response.tool_calls:
        logger.info("Generate called tool: %s", response.tool_calls[0]["name"])
        return {"messages": [response]}

    logger.info("Generated answer (%d chars)", len(response.content))
    return {"generation": response.content, "messages": [AIMessage(content=response.content)]}


def react_node(state: AgentState, llm_with_tools):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _format_history(messages) -> str:
    lines = []
    for m in list(messages)[-6:-1]:
        role = getattr(m, "type", "unknown")
        content = getattr(m, "content", "")
        if content:
            lines.append(f"{role}: {content[:200]}")
    return "\n".join(lines) if lines else "No prior conversation."

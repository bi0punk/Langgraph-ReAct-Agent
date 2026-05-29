import logging

from langchain_core.tools import tool

from app.retriever import get_retriever

logger = logging.getLogger(__name__)


@tool
def retrieve_documents(query: str) -> str:
    """Search the FAISS vectorstore for information relevant to the query. Use this when you need to look up facts, procedures, or details from the documentation."""
    logger.info("Tool call: retrieve_documents(query='%s')", query)
    try:
        retriever = get_retriever()
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        result = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )
        return result
    except FileNotFoundError:
        return "Error: Vectorstore not found. Please run 'python -m app.ingest' first."
    except Exception as e:
        logger.exception("Error in retrieve_documents")
        return f"Error retrieving documents: {e}"

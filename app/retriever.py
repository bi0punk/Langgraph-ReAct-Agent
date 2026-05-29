import logging

from langchain_community.vectorstores import FAISS

from app.config import settings
from app.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_vectorstore = None


def get_retriever():
    global _vectorstore
    if _vectorstore is None:
        path = settings.vectorstore_path
        if not path.exists():
            logger.error("Vectorstore not found at %s — run ingest.py first", path)
            raise FileNotFoundError(f"Vectorstore not found at {path}")

        embeddings = get_embeddings()
        _vectorstore = FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore.as_retriever(search_kwargs={"k": settings.rag_top_k})


def invalidate_cache():
    global _vectorstore
    _vectorstore = None

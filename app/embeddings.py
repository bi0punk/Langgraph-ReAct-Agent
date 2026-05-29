from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from app.config import settings

_instance = None


def get_embeddings():
    global _instance
    if _instance is None:
        _instance = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
        )
    return _instance

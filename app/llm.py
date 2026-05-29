from langchain_openai import ChatOpenAI

from app.config import settings

_instance = None


def get_llm():
    global _instance
    if _instance is None:
        _instance = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key or None,
            base_url=settings.openai_base_url or None,
            temperature=0,
            streaming=True,
        )
    return _instance

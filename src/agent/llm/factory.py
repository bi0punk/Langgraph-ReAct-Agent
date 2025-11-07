# src/agent/llm/factory.py
# ------------------------------------------------------------
# Fábrica de modelos LLM y Embeddings.
# Aquí se decide qué modelo usar, su temperatura, timeouts, etc.
# Se mantiene independiente del grafo y del RAG.
# ------------------------------------------------------------

import os
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def make_chat_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: int = 60,
):
    """
    Crea un modelo de chat listo para usar en el agente.

    Args:
        model (str): nombre del modelo (default: env OPENAI_MODEL o "gpt-4o-mini")
        temperature (float): creatividad del modelo
        timeout (int): timeout de request en segundos

    Returns:
        ChatOpenAI: cliente del modelo configurado
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Validación de API Key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("🚨 Falta variable de entorno OPENAI_API_KEY")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        timeout=timeout,
    )


def make_embeddings(
    model: Optional[str] = None,
):
    """
    Crea el modelo de embeddings para RAG.

    Args:
        model (str): nombre del embedding model (default: env OPENAI_EMBEDDINGS_MODEL o "text-embedding-3-small")

    Returns:
        OpenAIEmbeddings: embedding model listo para vectorstore
    """
    model = model or os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")

    # Validación de API Key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("🚨 Falta variable de entorno OPENAI_API_KEY")

    return OpenAIEmbeddings(model=model)


__all__ = ["make_chat_llm", "make_embeddings"]

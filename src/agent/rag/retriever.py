# src/agent/rag/retriever.py
# ------------------------------------------------------------
# Construcción y carga de índice FAISS para RAG.
# Expone:
#   - build_or_load_vectorstore(...)
#   - build_retriever(...)
# ------------------------------------------------------------

from __future__ import annotations
import os
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from .loaders import load_documents


def build_or_load_vectorstore(
    manuals_dir: str,
    index_dir: str,
    *,
    embedding_model: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> FAISS:
    """
    Crea o carga un índice FAISS desde 'index_dir' usando documentos en 'manuals_dir'.

    - Si existe index.faiss en index_dir → se carga.
    - Si no existe → se leen documentos, se trocean y se construye el índice.

    Args:
        manuals_dir: ruta al directorio de manuales (.txt, .md, .pdf)
        index_dir:   ruta al directorio donde vivirá el índice FAISS
        embedding_model: nombre del modelo de embeddings (OpenAI)
        chunk_size, chunk_overlap: parámetros del splitter

    Returns:
        FAISS (vectorstore) listo para .as_retriever(...)
    """
    os.makedirs(index_dir, exist_ok=True)
    emb = OpenAIEmbeddings(model=embedding_model or os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"))

    index_path = os.path.join(index_dir, "index.faiss")
    if os.path.exists(index_path):
        # Cargar índice existente
        return FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)

    # Construir desde cero
    docs = load_documents(manuals_dir)
    if not docs:
        # índice vacío (evita romper flujo si aún no hay manuales)
        return FAISS.from_texts(["(índice vacío)"], emb)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs) if docs else []

    vs = FAISS.from_documents(chunks, emb) if chunks else FAISS.from_texts(["(índice vacío)"], emb)
    vs.save_local(index_dir)
    return vs


def build_retriever(
    manuals_dir: str,
    index_dir: str,
    *,
    k: int = 4,
    embedding_model: Optional[str] = None,
):
    """
    Devuelve un retriever (vectorstore.as_retriever) listo para usar en el nodo RAG
    o en una tool `manual_search`.

    Args:
        manuals_dir: directorio de manuales
        index_dir:   directorio del índice
        k:           top-k de documentos a recuperar
        embedding_model: override opcional del modelo de embeddings

    Returns:
        retriever callable: retriever.invoke(query) -> List[Document]
    """
    vs = build_or_load_vectorstore(
        manuals_dir=manuals_dir,
        index_dir=index_dir,
        embedding_model=embedding_model,
    )
    return vs.as_retriever(search_kwargs={"k": k})

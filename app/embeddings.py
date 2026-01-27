#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from langchain_community.embeddings import HuggingFaceEmbeddings

# Modelo embeddings (CPU friendly)
EMBEDDING_MODEL_NAME = "intfloat/e5-small"

_embedding_instance = None


def get_embeddings():
    """
    Singleton embedding loader to avoid reloading model multiple times.
    """
    global _embedding_instance

    if _embedding_instance is None:
        print(f"🧠 Loading embeddings model: {EMBEDDING_MODEL_NAME}")

        _embedding_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    return _embedding_instance

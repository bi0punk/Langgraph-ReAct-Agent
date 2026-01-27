#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# =========================
# PROJECT ROOT
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =========================
# DATA PATHS
# =========================
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "faiss_index"

LOGS_DIR = PROJECT_ROOT / "logs"

# =========================
# EMBEDDINGS CONFIG
# =========================
EMBEDDING_MODEL_NAME = "intfloat/e5-small"

# =========================
# LLM CONFIG (GGUF / llama.cpp)
# =========================
LLAMA_CPP_PATH = Path("/home/drbash/Documentos/auto-local-llm/llama.cpp")

MODELS_DIR = Path("/home/drbash/Documentos/auto-local-llm/models")

DEFAULT_GGUF_MODEL = MODELS_DIR / "Qwen2.5-14B-Instruct.Q4_K_M.gguf"

# =========================
# PERFORMANCE LIMITS (CPU SAFE)
# =========================
MAX_CPU_THREADS = 4
MAX_CONTEXT_TOKENS = 4096
MAX_BATCH_SIZE = 8

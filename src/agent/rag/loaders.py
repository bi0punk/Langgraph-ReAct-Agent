# src/agent/rag/loaders.py
# ------------------------------------------------------------
# Carga documentos desde un directorio (txt, md, pdf)
# y devuelve una lista de Document de LangChain.
# ------------------------------------------------------------

from __future__ import annotations
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document 
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader


def load_documents(manuals_dir: str) -> List[Document]:
    """
    Carga .txt, .md y .pdf desde 'manuals_dir' recursivamente.

    Devuelve:
        List[Document] con metadata útil (source, page para PDF).
    """
    os.makedirs(manuals_dir, exist_ok=True)
    docs: List[Document] = []

    # TXT / MD
    if Path(manuals_dir).exists():
        txt_loader = DirectoryLoader(
            manuals_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
            recursive=True,
            silent_errors=True,
        )
        md_loader = DirectoryLoader(
            manuals_dir,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
            recursive=True,
            silent_errors=True,
        )
        docs.extend(txt_loader.load())
        docs.extend(md_loader.load())

        # PDF (PyPDFLoader maneja metadata de páginas)
        for root, _, files in os.walk(manuals_dir):
            for f in files:
                if f.lower().endswith(".pdf"):
                    path = os.path.join(root, f)
                    try:
                        docs.extend(PyPDFLoader(path).load())
                    except Exception as e:
                        print(f"[WARN] No se pudo cargar PDF {path}: {e}")

    return docs

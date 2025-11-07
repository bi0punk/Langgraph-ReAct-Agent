#!/usr/bin/env python3
# scripts/build_index.py
# ------------------------------------------------------------
# Construye o reconstruye el índice FAISS para el RAG
# leyendo documentos desde --manuals-dir y guardando en --index-dir.
#
# Uso:
#   # como script
#   python scripts/build_index.py --manuals-dir ./data/manuales --index-dir ./data/rag_index --rebuild
#
#   # como módulo (recomendado para evitar problemas de PYTHONPATH)
#   python -m scripts.build_index --manuals-dir ./data/manuales --index-dir ./data/rag_index
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Cargar .env si existe (para OPENAI_API_KEY, etc.)
try:
    from dotenv import load_dotenv  # noqa
    load_dotenv()
except Exception:
    pass


# Asegura que el proyecto esté en sys.path si se ejecuta como script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.rag.retriever import build_or_load_vectorstore  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Construye el índice FAISS para RAG.")
    p.add_argument("--manuals-dir", default="./data/manuales", help="Directorio con manuales (.txt, .md, .pdf)")
    p.add_argument("--index-dir", default="./data/rag_index", help="Directorio destino del índice FAISS")
    p.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"),
                   help="Modelo de embeddings (OpenAI). Por defecto: env OPENAI_EMBEDDINGS_MODEL o text-embedding-3-small")
    p.add_argument("--chunk-size", type=int, default=1000, help="Tamaño de chunk para el splitter")
    p.add_argument("--chunk-overlap", type=int, default=150, help="Solape de chunks")
    p.add_argument("--rebuild", action="store_true", help="Borra el índice existente antes de construir")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Falta la variable de entorno OPENAI_API_KEY")
        sys.exit(1)

    manuals_dir = Path(args.manuals_dir)
    index_dir = Path(args.index_dir)

    if args.rebuild and index_dir.exists():
        print(f"🧹 Borrando índice existente en: {index_dir}")
        shutil.rmtree(index_dir, ignore_errors=True)

    print(f"📚 Manuales: {manuals_dir}")
    print(f"📦 Índice:   {index_dir}")
    print(f"🔤 Emb:      {args.embedding_model}")
    print(f"🔪 Chunk:    size={args.chunk_size} overlap={args.chunk_overlap}")

    vs = build_or_load_vectorstore(
        manuals_dir=str(manuals_dir),
        index_dir=str(index_dir),
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Guardado ya lo hace build_or_load_vectorstore cuando crea desde cero.
    # Si cargó existente, simplemente confirmamos.
    print("✅ Índice listo.")
    print(f"📁 Archivos en {index_dir}:")
    for p in sorted(index_dir.rglob("*")):
        print("   -", p.relative_to(index_dir))


if __name__ == "__main__":
    main()

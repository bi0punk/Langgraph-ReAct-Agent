# pytest: skip-file

#!/usr/bin/env python3
# scripts/test_tools.py
# ------------------------------------------------------------
# Pruebas simples para las tools:
#   - calculator
#   - now_time
#   - manual_search (requiere índice o se creará si no existe)
#
# Uso:
#   python -m scripts.test_tools --q "consulta para manual_search"
#   python scripts/test_tools.py --q "otra consulta"
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Asegura que el proyecto esté en sys.path si se ejecuta como script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.tools import calculator, now_time, manual_search, set_retriever  # noqa: E402
from src.agent.rag.retriever import build_retriever  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Prueba de tools.")
    p.add_argument("--manuals-dir", default="./data/manuales", help="Directorio con manuales (.txt, .md, .pdf)")
    p.add_argument("--index-dir", default="./data/rag_index", help="Directorio del índice FAISS")
    p.add_argument("--k", type=int, default=4, help="Top-K de manual_search")
    p.add_argument("--q", default="introducción", help="Consulta para manual_search")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY no está definida. calculator/now_time funcionan, manual_search necesita embeddings.")
        # seguimos; manual_search fallará si intenta embeddings sin API key

    # --- Tools sin LLM ---
    print("\n➗ Test calculator:")
    print('  calculator("3*(4+5)") ->', calculator.run("3*(4+5)"))
    print('  calculator("sqrt(81)") ->', calculator.run("sqrt(81)"))

    print("\n🕒 Test now_time:")
    print("  now_time() ->", now_time.run(""))

    # --- manual_search (requiere retriever) ---
    print("\n📚 Test manual_search:")
    try:
        retriever = build_retriever(args.manuals_dir, args.index_dir, k=args.k)
        set_retriever(retriever)
        print("  manual_search(...) ->")
        print(manual_search.run(args.q))
    except Exception as e:
        print("  ⚠️ manual_search no pudo ejecutarse:", e)

    print("\n✅ Fin de pruebas.")


if __name__ == "__main__":
    main()



# # 1) Variables (si usas OpenAI)
# export OPENAI_API_KEY="sk-..."
# export OPENAI_EMBEDDINGS_MODEL="text-embedding-3-small"   # opcional

# # 2) Construir (o reconstruir) el índice
# python -m scripts.build_index --manuals-dir ./data/manuales --index-dir ./data/rag_index --rebuild

# # 3) Probar tools
# python -m scripts.test_tools --q "cómo configurar X en el manual"

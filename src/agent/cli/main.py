# src/agent/cli/main.py
# -------------------------------------------------------------------
# CLI del agente (terminal).
#
# Requisitos previos del proyecto:
#   - src/agent/app.py debe exponer get_app() -> LangGraph "compiled app"
#   - config/settings.yaml y config/logging.yaml (opcional pero recomendado)
#   - OPENAI_API_KEY en el entorno (si usas OpenAI)
#
# Uso:
#   python -m src.agent.cli.main
#   python -m src.agent.cli.main --thread-id demo --no-banner
#
# Salir:
#   escribir: salir | exit | quit
# -------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import sys
import logging
from logging.config import dictConfig
from pathlib import Path

# --- Carga .env (opcional) ---
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# --- Logging desde config (si existe) ---
def _setup_logging() -> None:
    cfg_path = Path(os.getenv("LOGGING_CONFIG", "./config/logging.yaml"))
    if cfg_path.exists():
        try:
            import yaml  # type: ignore

            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            dictConfig(cfg)
            return
        except Exception as e:
            print(f"[WARN] No se pudo aplicar logging.yaml ({e}); usando logging básico.")
    # fallback básico
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


_setup_logging()
log = logging.getLogger("agent.cli")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLI del agente LangGraph (act→observe→reason)")
    p.add_argument(
        "--thread-id",
        default=os.getenv("THREAD_ID", "terminal-1"),
        help="Identificador lógico de la conversación para conservar memoria.",
    )
    p.add_argument(
        "--no-banner",
        action="store_true",
        help="No mostrar banner inicial.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Verificación mínima (si usas OpenAI)
    if not os.getenv("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY no está definida. Si usas OpenAI, debes configurarla.")

    # Import diferido para no romper si aún no implementas app.py
    try:
        from src.agent.app import get_app  # tipo: ()->Any
    except Exception as e:
        log.error("No se pudo importar get_app() desde src.agent.app. ¿Ya implementaste app.py?")
        log.debug("Detalle de import error", exc_info=True)
        print("❌ Falta implementar src/agent/app.py con una función get_app().")
        return 1

    try:
        app = get_app()
    except Exception as e:
        log.error("Error construyendo la app del agente: %s", e, exc_info=True)
        print("❌ Error inicializando el agente. Revisa configuración y dependencias.")
        return 2

    if not args.no_banner:
        print("🤖 Agente (act→observe→reason) listo. Escribe 'salir' para terminar.")
        print(f"🧵 thread_id: {args.thread_id}")

    # Helpers para mensajes
    from langchain_core.messages import HumanMessage, AIMessage
    from src.agent.memory.memory import thread_config

    while True:
        try:
            q = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAdiós!")
            break

        if not q:
            continue
        if q.lower() in {"salir", "exit", "quit"}:
            print("Adiós!")
            break

        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=q)], "steps": 0},
                config=thread_config(args.thread_id, metadata={"channel": "cli"}),
            )
        except Exception as e:
            log.error("Fallo ejecutando el grafo: %s", e, exc_info=True)
            print("❌ Ocurrió un error al procesar tu mensaje. Revisa logs.")
            continue

        ai_msgs = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
        reply = ai_msgs[-1].content if ai_msgs else "(sin respuesta)"
        print(f"\nAsistente: {reply}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

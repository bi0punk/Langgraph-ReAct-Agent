import logging
import os

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from app.config import settings
from app.graph import create_pipeline
from app.logging_config import setup_logging

logger = logging.getLogger(__name__)

G = "\033[32m"
B = "\033[34m"
Y = "\033[33m"
R = "\033[31m"
M = "\033[35m"
C = "\033[36m"
RESET = "\033[0m"


class TokenPrinter(BaseCallbackHandler):
    def __init__(self):
        self.seen_first = False

    def on_llm_new_token(self, token: str, **kwargs):
        if not self.seen_first:
            print(f"\n{B}Assistant:{RESET} ", end="", flush=True)
            self.seen_first = True
        print(token, end="", flush=True)


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def print_help():
    print(f"""
{M}╔══════════════════════════════════════════════╗{RESET}
{M}║     LangGraph RAG Pipeline — Comandos        ║{RESET}
{M}╠══════════════════════════════════════════════╣{RESET}
{M}║{RESET}  /new      Nueva conversación (nuevo thread) {M}║{RESET}
{M}║{RESET}  /history  Muestra el historial del thread  {M}║{RESET}
{M}║{RESET}  /clear    Limpia la pantalla                {M}║{RESET}
{M}║{RESET}  /mode     Muestra la configuración actual   {M}║{RESET}
{M}║{RESET}  /help     Muestra esta ayuda                {M}║{RESET}
{M}║{RESET}  /exit     Salir                             {M}║{RESET}
{M}╚══════════════════════════════════════════════╝{RESET}
    """)


def print_mode():
    print(f"{C}╔══ Configuración actual ══╗{RESET}")
    print(f"{C}║{RESET} RAG_MODE:      {Y}{settings.rag_mode}{RESET}")
    print(f"{C}║{RESET} LLM:           {Y}{settings.openai_model}{RESET}")
    if settings.openai_base_url:
        print(f"{C}║{RESET} Base URL:      {Y}{settings.openai_base_url}{RESET}")
    print(f"{C}║{RESET} Embeddings:    {Y}{settings.embedding_model}{RESET}")
    print(f"{C}║{RESET} Top-K:         {Y}{settings.rag_top_k}{RESET}")
    print(f"{C}║{RESET} Max rewrites:  {Y}{settings.max_rewrites}{RESET}")
    print(f"{C}║{RESET} Checkpointer:  {Y}{settings.checkpointer}{RESET}")
    print(f"{C}║{RESET} Thread ID:     {Y}{settings.thread_id}{RESET}")
    print(f"{C}╚═════════════════════════╝{RESET}")


def stream_answer(agent, question: str, thread_id: str):
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [TokenPrinter()],
    }

    initial_state = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "documents": [],
        "generation": "",
        "rewrite_count": 0,
    }

    events = agent.stream(initial_state, config)

    for event in events:
        for node_name, output in event.items():
            if node_name == "tools":
                print(f"\n{Y}  🔍 Search complete{RESET}")
            elif node_name == "rewrite":
                print(f"\n{Y}  🔄 Rewriting query...{RESET}")
            elif node_name == "retrieve":
                count = len(output.get("documents", []))
                if count > 0:
                    print(f"\n{C}  📄 {count} document(s) retrieved{RESET}")
            elif node_name == "grade":
                g = output.get("grade", "")
                if g == "relevant":
                    print(f"\n{G}  ✓ Documents relevant{RESET}")
                elif g == "not_relevant":
                    print(f"\n{R}  ✗ Documents not relevant{RESET}")


def show_history(agent, thread_id: str):
    try:
        checkpoint = agent.checkpointer.get({"configurable": {"thread_id": thread_id}})
        if checkpoint and "channel_values" in checkpoint and checkpoint["channel_values"].get("messages"):
            messages = checkpoint["channel_values"]["messages"]
            print(f"\n{C}╔══ Historial del thread: {thread_id} ══╗{RESET}")
            for msg in messages:
                role = getattr(msg, "type", "unknown")
                content = getattr(msg, "content", "")
                if role == "human":
                    print(f"{G}You:{RESET} {content}")
                elif role == "ai":
                    print(f"{B}Assistant:{RESET} {content[:200]}{'...' if len(content) > 200 else ''}")
            print(f"{C}╚{'═' * 50}╝{RESET}")
        else:
            print(f"{Y}No conversation history for this thread.{RESET}")
    except Exception as e:
        logger.warning("Could not load history: %s", e)
        print(f"{Y}No history available.{RESET}")


def main():
    setup_logging()
    pipeline = create_pipeline()
    thread_id = settings.thread_id

    clear()
    print(f"{M}╔══════════════════════════════════════════════╗{RESET}")
    print(f"{M}║       LangGraph RAG Pipeline v2             ║{RESET}")
    print(f"{M}║       RAG_MODE: {settings.rag_mode:<24}║{RESET}")
    print(f"{M}║       Model: {settings.openai_model:<29}║{RESET}")
    print(f"{M}║       Thread: {thread_id:<28}║{RESET}")
    print(f"{M}║       Type /help for commands               ║{RESET}")
    print(f"{M}╚══════════════════════════════════════════════╝{RESET}")

    while True:
        try:
            q = input(f"\n{G}Question >{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            continue

        if q.startswith("/"):
            cmd = q.lower()
            if cmd in ("/exit", "/quit"):
                break
            elif cmd == "/clear":
                clear()
            elif cmd == "/help":
                print_help()
            elif cmd == "/mode":
                print_mode()
            elif cmd == "/history":
                show_history(pipeline, thread_id)
            elif cmd == "/new":
                import uuid
                thread_id = str(uuid.uuid4())
                print(f"{C}New thread: {thread_id}{RESET}")
            else:
                print(f"{R}Unknown command: {q}{RESET}")
            continue

        try:
            stream_answer(pipeline, q, thread_id)
            print()
        except Exception as e:
            logger.exception("Error processing question")
            print(f"\n{R}Error: {e}{RESET}")


if __name__ == "__main__":
    main()

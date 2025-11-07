# src/agent/prompts/templates.py
# ------------------------------------------------------------
# Construcción del prompt del agente usando un "system prompt"
# externo (system.txt) + el historial de mensajes.
#
# Este módulo genera `ChatPromptTemplate` usado por el grafo.
# ------------------------------------------------------------

import os
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def load_system_prompt() -> str:
    """
    Lee el prompt del sistema desde system.txt.
    Si no existe, levanta un error claro.
    """
    path = Path(__file__).parent / "system.txt"
    if not path.exists():
        raise FileNotFoundError(f"❌ No se encontró system.txt en: {path}")
    return path.read_text(encoding="utf-8").strip()


def get_prompt() -> ChatPromptTemplate:
    """
    Construye el prompt del agente:
      - Mensaje system tomado de `system.txt`
      - Historial dinámico `MessagesPlaceholder("messages")`
      - Otras variables se pueden agregar si se desea (ej: rag_ctx, user_profile, etc.)
    """
    system_text = load_system_prompt()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            MessagesPlaceholder(variable_name="messages"),   # historial conversacional acumulado
        ]
    )
    return prompt


__all__ = ["get_prompt", "load_system_prompt"]

# src/agent/tools/calculator.py
# ------------------------------------------------------------
# Tool: calculator
# Evalúa expresiones matemáticas simples de forma segura.
# ------------------------------------------------------------

from langchain_core.tools import tool


@tool("calculator")
def calculator(expression: str) -> str:
    """
    Evalúa una expresión matemática segura usando math.
    Ejemplos:
      "2+2"
      "sqrt(9)"
      "3 * (4 + 5)"
    """
    import math

    allowed = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
    allowed["__builtins__"] = {}

    try:
        result = eval(expression, allowed, {})
    except Exception as e:
        return f"Error evaluando la expresión: {e}"

    return str(result)

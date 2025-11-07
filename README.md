
# Documentación del Proyecto: Agente LangGraph + RAG + Tools

## 1. Descripción General
Este proyecto implementa un agente conversacional con arquitectura **act → observe → reason**, 
memoria, herramientas externas (tools) y búsqueda en manuales mediante RAG (Retrieval-Augmented Generation).

El agente:
- Responde preguntas generales usando un modelo LLM.
- Cuando corresponde, busca información en manuales internos.
- Decide acciones internamente y usa herramientas cuando es necesario.
- Mantiene historial conversacional mediante canales acumulativos (`add_messages`).

---

## 2. Características Principales

| Componente | Descripción |
|-----------|-------------|
| **LangGraph** | Orquesta el flujo act → observe → reason. |
| **RAG** | Recupera información de manuales mediante embeddings y FAISS. |
| **Tools** | Calculadora y hora actual (extendibles). |
| **Memoria** | Se puede usar memoria en RAM o SQLite. |
| **CLI** | Interfaz interactiva en terminal. |
| **API (opcional)** | Servidor HTTP con endpoints REST. |

---

## 3. Flujo de Ejecución (Ciclo del Agente)

```
Usuario → rag_prepare → act → tool_node (si aplica) → observe → reason → Respuesta final
```

- `rag_prepare`: Si la consulta es relevante para los manuales → inyecta contexto.
- `act`: El LLM decide si llamar a tools o responder directamente.
- `observe`: Si hubo tool, se recoge su resultado.
- `reason`: Redacta la respuesta final en español.

---

## 4. Estructura del Proyecto

```
project-root/
├── config/
│   ├── settings.yaml       # Configuración del agente
│   └── logging.yaml        # Logging opcional
├── data/
│   ├── manuales/           # Manuales fuente (txt/md/pdf)
│   └── rag_index/          # Índice vectorial generado
├── scripts/
│   └── build_index.py      # Construcción del índice RAG
├── src/
│   └── agent/
│       ├── app.py          # Ensambla el agente
│       ├── cli/main.py     # Interfaz CLI
│       ├── llm/factory.py  # Modelos LLM y embeddings
│       ├── rag/            # Carga documentos + RAG
│       ├── tools/          # Herramientas externas
│       ├── graph/          # Grafo act → observe → reason
│       ├── prompts/        # Prompt del agente
│       └── state.py        # Definición del estado del agente
└── tests/                  # Pruebas unitarias
```

---

## 5. Instalación

```bash
git clone <repo>
cd proyecto
python -m venv ve
source ve/bin/activate
pip install -r requirements.txt
```

Crear archivo `.env`:
```
OPENAI_API_KEY=tu_api_key
```

---

## 6. Construcción del Índice RAG

```bash
python -m scripts.build_index --rebuild
```

---

## 7. Ejecutar CLI

```bash
python -m src.agent.cli.main --no-banner
```

Ejemplo:

```
Tú: quien creó linux
Asistente: Linus Torvalds es...
```

---

## 8. Agregar Manuales

Coloca archivos `.txt`, `.md` y `.pdf` dentro de:
```
data/manuales/
```
Luego reconstruye el índice.

---

## 9. Extender con Nuevas Tools

Ejemplo: agregar clima:

```python
from langchain.tools import tool

@tool
def weather(city: str):
    "Devuelve estado del tiempo"
```

Luego añadir en `app.py` dentro de la lista `tools=[...]`.

---

## 10. Pruebas

```bash
pytest -q
```

---

## 11. Licencia
Uso libre para desarrollo interno.

---

Fin de la documentación.
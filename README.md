# Langgraph-ReAct-Agent

Corrective RAG (Retrieval-Augmented Generation) pipeline built with LangGraph. Implements a ReAct agent pattern with vector search (FAISS), document ingestion, and LLM integration via CLI interface.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)

## Tabla de Contenidos

- [Características](#características)
- [Stack](#stack)
- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Tests](#tests)
- [Configuración](#configuración)
- [Datos](#datos)
- [Limitaciones / Roadmap](#limitaciones--roadmap)
- [Licencia](#licencia)

## Características

- Corrective RAG pipeline con LangGraph y ReAct agent pattern
- Indexación de documentos con FAISS (Facebook AI Similarity Search)
- Embeddings con HuggingFace Sentence-Transformers
- CLI interactiva para consultas al agente
- Arquitectura modular: graph, nodes, tools, retriever separados

## Stack

- Python 3.10+, LangGraph, LangChain, HuggingFace, FAISS, Sentence-Transformers

## Arquitectura

```
Langgraph-ReAct-Agent/
├── app/
│   ├── cli.py              # CLI interactiva
│   ├── ingest.py           # Ingesta de documentos
│   ├── graph/              # Definición del grafo LangGraph
│   ├── nodes/              # Nodos del pipeline
│   ├── tools/              # Herramientas del agente
│   └── retriever/          # Módulo de retrieval (FAISS)
├── data/                   # Documentos fuente
├── vectorstore/            # Índices FAISS persistentes
├── scripts/                # Scripts auxiliares
├── logs/                   # Logs de ejecución
├── tests/
├── requirements.txt
├── pyproject.toml
└── .env.sample
```

## Requisitos

- Python 3.10+
- Acceso a modelo LLM (HuggingFace, OpenAI)

## Instalación

```bash
git clone https://github.com/tu-usuario/Langgraph-ReAct-Agent.git
cd Langgraph-ReAct-Agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# 1. Ingestar documentos en el vectorstore
python -m app.ingest

# 2. Iniciar CLI interactiva
python -m app.cli

# Ejemplo en CLI:
# > ¿Cuál es el procedimiento para X?
```

## Tests

```bash
pip install pytest ruff
pytest -q
ruff check .
```

## Configuración

Variables de entorno (ver `.env.sample`):

| Variable            | Descripción                           |
|---------------------|---------------------------------------|
| `HUGGINGFACE_TOKEN` | Token de HuggingFace (opcional)       |
| `OPENAI_API_KEY`    | API key de OpenAI (si se usa)         |
| `EMBEDDING_MODEL`   | Modelo de embeddings (default: all-MiniLM-L6-v2) |

## Datos

- Documentos fuente en `data/` (formato .txt, .pdf, .md)
- Índices FAISS generados se almacenan en `vectorstore/`
- Los vectores se persisten para reutilización sin re-ingesta

## Limitaciones / Roadmap

- [ ] Soporte para más fuentes de datos (web scraping, APIs)
- [ ] Evaluación de calidad de retrieval (MRR, NDCG)
- [ ] Interfaz web con Streamlit
- [ ] Auto-refresh de índices con nuevos documentos

## Licencia

MIT

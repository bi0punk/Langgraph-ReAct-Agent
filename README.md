# Langgraph-ReAct-Agent

Corrective RAG (Retrieval-Augmented Generation) pipeline built with LangGraph. Implements a ReAct agent pattern with vector search (FAISS), document ingestion, and LLM integration via a CLI interface.

## Stack

Python 3.10+, LangGraph, LangChain, HuggingFace, FAISS, Sentence-Transformers

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Ingest documents
python -m app.ingest

# Run the CLI
python -m app.cli
```

## Structure

```
Langgraph-ReAct-Agent/
├── app/
│   ├── cli.py
│   ├── ingest.py
│   ├── graph/
│   ├── nodes/
│   ├── tools/
│   └── retriever/
├── data/
├── vectorstore/
├── tests/
├── pyproject.toml
├── requirements.txt
└── .env.sample
```

## License

MIT

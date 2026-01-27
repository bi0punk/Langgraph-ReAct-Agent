#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.embeddings import get_embeddings
from app.config import DOCS_DIR, VECTORSTORE_DIR

def load_documents():
    docs = []
    docs_path = Path(DOCS_DIR)

    if not docs_path.exists():
        raise RuntimeError(f"Docs directory not found: {DOCS_DIR}")

    for file in docs_path.glob("*"):
        if file.suffix.lower() == ".pdf":
            print(f"📄 Loading PDF: {file}")
            loader = PyPDFLoader(str(file))
            docs.extend(loader.load())

        elif file.suffix.lower() in [".txt", ".md"]:
            print(f"📄 Loading text: {file}")
            loader = TextLoader(str(file))
            docs.extend(loader.load())

    return docs


def ingest():
    print("🚀 Starting ingestion pipeline...")

    docs = load_documents()

    if not docs:
        raise RuntimeError("❌ No documents loaded. Check data/docs content.")

    print(f"✅ Loaded {len(docs)} raw pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        raise RuntimeError("❌ No chunks generated")

    print(f"🧩 Generated {len(chunks)} chunks")

    embedding_model = get_embeddings()

    texts = [c.page_content for c in chunks]

    print("🧠 Generating embeddings...")
    vectors = embedding_model.embed_documents(texts)

    print("📦 Building FAISS index...")
    db = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=embedding_model
    )

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(VECTORSTORE_DIR))

    print(f"✅ Vectorstore saved to: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    ingest()

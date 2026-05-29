import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def load_pdfs(docs_path: Path) -> list:
    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", docs_path)
        return []

    docs = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            docs.extend(pages)
            logger.info("Loaded %s — %d pages", pdf_file.name, len(pages))
        except Exception as e:
            logger.error("Failed to load %s: %s", pdf_file.name, e)
    return docs


def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    return splitter.split_documents(docs)


def build_vectorstore(split_docs: list):
    embeddings = get_embeddings()
    logger.info("Building FAISS index from %d chunks...", len(split_docs))
    return FAISS.from_documents(split_docs, embeddings)


def save_vectorstore(vectorstore, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(path))
    logger.info("Vectorstore saved to %s", path)


def main():
    docs_path = settings.docs_path
    vectorstore_path = settings.vectorstore_path

    logger.info("Ingesting PDFs from %s", docs_path)
    docs_path.mkdir(parents=True, exist_ok=True)

    docs = load_pdfs(docs_path)
    if not docs:
        logger.error("No documents loaded — aborting")
        return

    split_docs = split_documents(docs)
    logger.info("Created %d chunks", len(split_docs))

    vectorstore = build_vectorstore(split_docs)
    save_vectorstore(vectorstore, vectorstore_path)
    logger.info("Ingestion complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

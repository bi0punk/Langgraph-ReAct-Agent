from langchain_community.vectorstores import FAISS
from app.embeddings import embedding_model
from app.config import VECTOR_PATH, RETRIEVER_K

db = FAISS.load_local(VECTOR_PATH, embedding_model)

def retrieve(query):
    return db.similarity_search(query, k=RETRIEVER_K)

from app.llm_cli import run_llama
from app.retriever import retrieve


PROMPT = """
You are a technical assistant.
Answer using ONLY the context.

Context:
{context}

Question:
{question}

Answer concisely:
"""


def generate_rag_answer(question):
    docs = retrieve(question)

    context = "\n".join(d.page_content[:700] for d in docs)

    prompt = PROMPT.format(
        context=context,
        question=question
    )

    return run_llama(prompt)

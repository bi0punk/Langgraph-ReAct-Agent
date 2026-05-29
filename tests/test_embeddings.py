from app.embeddings import get_embeddings


def test_get_embeddings():
    embeddings = get_embeddings()
    assert embeddings is not None
    assert hasattr(embeddings, "model_name")


def test_embed_documents():
    embeddings = get_embeddings()
    texts = ["Hello world", "Test sentence"]
    result = embeddings.embed_documents(texts)
    assert len(result) == 2
    assert len(result[0]) > 0
    assert len(result[1]) > 0


def test_embed_query():
    embeddings = get_embeddings()
    result = embeddings.embed_query("Test query")
    assert len(result) > 0

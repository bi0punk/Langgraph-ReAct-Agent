from app.graph import create_pipeline


def test_create_pipeline():
    pipeline = create_pipeline()
    assert pipeline is not None


def test_pipeline_type():
    pipeline = create_pipeline()
    assert hasattr(pipeline, "invoke") or hasattr(pipeline, "stream")

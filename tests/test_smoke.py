"""Smoke tests para Langgraph-ReAct-Agent (sin deps pesadas)."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_settings_load_defaults():
    """Settings debe instanciar con defaults."""
    from app.config import Settings

    s = Settings()
    assert s.rag_top_k >= 1
    assert s.rag_mode in {"none", "auto", "both"}
    assert s.checkpointer in {"memory", "sqlite"}


def test_create_pipeline_shape():
    """create_pipeline debe retornar un objeto con invoke/stream (no requiere LLM real)."""
    from app.graph import create_pipeline

    pipeline = create_pipeline()
    assert pipeline is not None
    assert hasattr(pipeline, "invoke") or hasattr(pipeline, "stream")


def test_no_pdf_tracked():
    """El manual.pdf no debe commitearse."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True, check=True
    ).stdout
    pdfs = [l for l in tracked.splitlines() if l.endswith(".pdf")]
    assert not pdfs, f"PDFs commiteados: {pdfs}"


@pytest.fixture
def settings():
    from app.config import Settings
    return Settings()


def test_env_sample_exists():
    assert (Path(__file__).resolve().parent.parent / ".env.sample").exists()
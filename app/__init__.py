def __getattr__(name):
    if name == "cli_main":
        from app.cli import main
        return main
    if name == "create_pipeline":
        from app.graph import create_pipeline
        return create_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["cli_main", "create_pipeline"]

"""Pipelined MLX runtime for Hibiki-Zero speech translation on staged model directories."""

__all__ = ["UPSTREAM", "W", "Model", "load", "resolve_weights_dir", "run", "translate"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from hibiki_mlx import pipeline

    return getattr(pipeline, name)

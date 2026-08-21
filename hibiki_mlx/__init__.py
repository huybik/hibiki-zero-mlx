"""Pipelined MLX runtime for Hibiki q4 and BF16 speech translation."""

__all__ = ["W", "load", "make_mimi", "resolve_weights_dir", "run"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from hibiki_mlx import pipeline

    return getattr(pipeline, name)

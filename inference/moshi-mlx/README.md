# Hibiki MLX model runtime

This is the minimal vendored subset of `moshi-mlx` used by Hibiki inference.
It contains the language model, streaming generator, transformer, KV cache,
conditioning, and sampling code. Audio encoding and decoding use `rustymimi`
through `hibiki_mlx.pipeline`.

Install it from the repository root:

```bash
pip install -e ./moshi-mlx
```

Use `main.py` for inference. This package is not a standalone CLI.

The code remains under the upstream MIT license in `LICENSE`.

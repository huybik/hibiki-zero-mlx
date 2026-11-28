# Hibiki MLX model runtime

This is the minimal vendored subset of `moshi-mlx` used by Hibiki inference.
It contains the language model, streaming generator, transformer, KV cache
and sampling code, cut to what Hibiki-Zero's models use: a model config that sets
any other upstream option (conditioners, cross-attention, layer scale, ...) is refused. Audio encoding and decoding use `rustymimi`
through `hibiki_mlx.pipeline`.

Install it from the repository root:

```bash
pip install -e ./inference/moshi-mlx
```

Use `inference/main.py` for inference. This package is not a standalone CLI.

The code remains under the upstream MIT license in `LICENSE`.

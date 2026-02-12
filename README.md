# Hibiki-Zero

Hibiki-Zero is a real-time and multilingual speech translation model.
It translates from French, Spanish, Portuguese and German to English: accurately, with low latency, high audio quality, and voice transfer.

<video controls width="400">
    <source src="https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/videos/clip_fr_translated.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>

[🤗 Hugging Face Model Card](https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16) | 
[⚙️ Tech report](https://kyutai.org/blog/2026-02-12-hibiki-zero) |
[📄 Paper](https://arxiv.org/abs/2602.11072) |
[🎧 More samples](https://huggingface.co/spaces/kyutai/hibiki-zero-samples)

## Requirements

Hibiki-Zero is a 3B-parameter model and requires an NVIDIA GPU to run: 8 GB VRAM should work, 12 GB is safe.

## Run the server

Hibiki-Zero comes with a server you can run to interact with Hibiki in real time. To run it, just use:

```python
uv run hibiki-zero serve [--gradio-tunnel]
```

Then go to the URL displayed to try out Hibiki-Zero.
The `--gradio-tunnel` flag will forward the server to a public URL that you can access from anywhere.

## Run inference

If you'd like to run Hibiki-Zero on existing audio files, run:

```python
uv run hibiki-zero generate [--file /path/to/my/audio.wav --file /path/to/another/audio.mp3]
```

Batch inference is supported, meaning you can run the model on multiple audio files at the same time.

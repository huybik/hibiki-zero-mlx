"""The client of the Qwen3-ASR server (`restart.sh asr`: `qwen-asr-serve` on :8100) and the error a transcript is
scored by ([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#filters)): YODAS's ingest hears its rows
with the language detected (yodas.py), voice.py its tries with the language forced.
"""
from __future__ import annotations

import io
import math
import time

import numpy as np

URL = "http://localhost:8100/v1/audio/transcriptions"
RATE = 16_000  # the rate the server hears
TIMEOUT_S = 60  # a request takes well under a second
MAX_TOKENS = 512  # a looping transcript comes back long (and scores broken) instead of running on


def wav_16k(pcm: np.ndarray, rate: int) -> bytes:
    """Mono PCM at `rate` -> the 16-bit 16 kHz WAV the server hears, resampled by scipy."""
    import soundfile as sf
    from scipy.signal import resample_poly

    g = math.gcd(RATE, rate)
    buf = io.BytesIO()
    sf.write(buf, resample_poly(pcm, RATE // g, rate // g).astype(np.float32), RATE, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def parse(raw: str) -> tuple[str, str]:
    """An unforced reply -> (language, transcript). The model writes `language <Name><asr_text><text>`,
    and `language None` for audio with no speech; a reply without the header is all
    transcript, its language unknown (""), as qwen_asr's own parser reads it. Clips over 30 s,
    which the server would split and join headers into, do not occur (rows are under 22.4 s)."""
    head, tag, text = raw.partition("<asr_text>")
    if not tag or not head.startswith("language "):
        return "", raw.strip()
    language = head.removeprefix("language ").strip()
    return ("", "") if language == "None" else (language, text.strip())


class Client:
    """One HTTP client to the server at `url`, `connections` requests at once. A request that fails to connect or
    send (`httpx.TransportError`) is sent again after `pause_s`, `tries` times in all; an HTTP error raises at once."""

    def __init__(self, url: str, connections: int, tries: int, pause_s: float) -> None:
        import httpx

        self.url, self.tries, self.pause_s = url, tries, pause_s
        self.http = httpx.Client(timeout=TIMEOUT_S, limits=httpx.Limits(max_connections=connections))

    def hear(self, wav: bytes, language: str | None = None) -> tuple[str, str]:
        """A clip's WAV (`wav_16k`) -> (language, transcript): forced to `language`, the reply as the server gives
        it; without, the language the model detected (`parse`)."""
        import httpx

        data = {"model": "asr"}
        if language:
            data |= {"language": language, "to_language": language}
        data["max_completion_tokens"] = MAX_TOKENS
        for attempt in range(self.tries):
            try:
                reply = self.http.post(self.url, files={"file": ("a.wav", wav, "audio/wav")}, data=data)
                break
            except httpx.TransportError:
                if attempt == self.tries - 1:
                    raise
                if self.pause_s:
                    time.sleep(self.pause_s)
        reply.raise_for_status()
        raw = reply.json()["text"]
        return (language, raw) if language else parse(raw)


def cer(text: str, asr: str) -> float:
    """A transcript's character error rate against its text, both as `filters.spoken` writes them (numbers as
    words); against no text, 1 for any transcript and 0 for none."""
    import jiwer

    from training.tranche.filters import spoken

    ref, hyp = " ".join(spoken(text)), " ".join(spoken(asr))
    return jiwer.cer(ref, hyp) if ref else float(bool(hyp))

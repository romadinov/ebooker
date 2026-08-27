"""Kokoro-82M backend (English and others). Apache 2.0, 82M params.

Measured on Old Man's War: 0.0% median CER across all six test cases and all
three voices, at RTF ~0.08 -- a 9.4-hour book in about 47 minutes on the M4 Max
CPU. No voice cloning; you get its built-in voices.

Output is quiet (peaks 0.33-0.68 in testing) rather than clipped, which the
mastering stage's RMS normalisation handles.

Requires espeak-ng for out-of-vocabulary words. The bundled espeakng-loader
ships a hard-coded CI path that does not exist, so ESPEAK_DATA_PATH must be set:
    brew install espeak-ng
    export ESPEAK_DATA_PATH=/opt/homebrew/opt/espeak-ng/share/espeak-ng-data
Its English G2P also needs the spaCy model en_core_web_sm.
"""

from __future__ import annotations

import os
import time

import numpy as np

from .base import Result

SAMPLE_RATE = 24000
REPO = "hexgrad/Kokoro-82M"

# Kokoro language codes, keyed by the EPUB's dc:language.
LANG_CODES = {
    "en": "a",      # American English ('b' for British)
    "es": "e", "fr": "f", "hi": "h", "it": "i", "pt": "p", "ja": "j", "zh": "z",
}

DEFAULT_VOICE = {"a": "am_michael", "b": "bm_george"}

_ESPEAK_HINTS = (
    "/opt/homebrew/opt/espeak-ng/share/espeak-ng-data",
    "/usr/lib/x86_64-linux-gnu/espeak-ng-data",
    "/usr/share/espeak-ng-data",
)


def _ensure_espeak() -> None:
    if os.environ.get("ESPEAK_DATA_PATH"):
        return
    for p in _ESPEAK_HINTS:
        if os.path.isdir(p):
            os.environ["ESPEAK_DATA_PATH"] = p
            return


class Kokoro:
    name = "kokoro"
    sample_rate = SAMPLE_RATE

    def __init__(self, voice: str | None = None, lang_code: str = "a",
                 speed: float = 1.0):
        _ensure_espeak()
        from kokoro import KPipeline

        self.lang_code = lang_code
        self.voice = voice or DEFAULT_VOICE.get(lang_code, "am_michael")
        self.speed = speed
        self.pipeline = KPipeline(lang_code=lang_code, repo_id=REPO)

    def synth(self, text: str, *, lang: str = "en", seed: int | None = None) -> Result:
        t0 = time.perf_counter()
        segs = list(self.pipeline(text, voice=self.voice, speed=self.speed))
        dt = time.perf_counter() - t0
        if not segs:
            return Result(np.zeros(0, dtype=np.float32), SAMPLE_RATE, 0.0, dt)
        audio = np.concatenate(
            [s.output.audio.cpu().numpy() for s in segs]).astype(np.float32)
        return Result(audio, SAMPLE_RATE, audio.size / SAMPLE_RATE, dt)

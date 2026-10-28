"""Stress-tuned Qwen3-TTS backend (Apache-2.0, Russian).

The model is siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf: Qwen3-TTS 1.7B Base
plus a LoRA trained to put Russian stress where a combining acute (U+0301)
says. Its card reports 100% of marks followed when they agree with the
dictionary and 96% when they contradict it. It was chosen by ear over Piper,
CosyVoice 3, Fish S2 Pro, Supertonic and ESpeech RL-V2.

Text treatment, also chosen by ear: ё restored and sparse acute marks from
silero (``to_acute_sparse``). Words the source already marks by hand keep their
own acute, so a deliberate за́мок / замо́к is never overridden.

Voice: cloned from a short reference clip plus its exact transcript. A
transcript that disagrees with its audio is what made earlier models babble,
so it is required, read from ``<reference>.txt`` when not passed.

Runaway guard: an autoregressive model can loop. A chunk far longer than its
text warrants is regenerated with another seed (Higgs once lost 10.6 GPU-hours
to exactly this); the pipeline's own verification is the second line.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import Result

REPO = "siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf"
ACUTE = "́"
MAX_SECONDS_PER_CHAR = 1 / 7.0   # measured reading rate is ~14 chars/second
MIN_CEILING_S = 4.0
ATTEMPTS = 3


def _acute_to_plus(word: str) -> str:
    """за́мок -> з+амок: the acute follows its vowel, '+' precedes it."""
    out: list[str] = []
    for ch in word:
        if ch == ACUTE and out:
            out.insert(len(out) - 1, "+")
        else:
            out.append(ch)
    return "".join(out)


class QwenStress:
    name = "qwen-stress"
    sample_rate = 24000

    def __init__(
        self,
        reference: str | Path,
        reference_text: str | Path | None = None,
        device: str = "auto",
        stress: str = "silero",
        model: str = REPO,
    ):
        import torch
        from qwen_tts import Qwen3TTSModel

        from ..device import pick as pick_device

        self._torch = torch
        self.device = pick_device(device)
        self.reference = str(reference)
        if reference_text is None:
            cand = Path(reference).with_suffix(".txt")
            if not cand.exists():
                raise FileNotFoundError(
                    f"{cand} missing: the reference clip needs its exact transcript")
            reference_text = cand
        if Path(str(reference_text)).exists():
            reference_text = Path(str(reference_text)).read_text(encoding="utf-8")
        self.reference_text = str(reference_text).replace("+", "").strip()

        self.stress = stress
        self._acc = None
        if stress == "silero":
            from ..ru_stress import SileroAccentuator
            self._acc = SileroAccentuator()

        dev = "cuda:0" if self.device == "cuda" else self.device
        self._model = Qwen3TTSModel.from_pretrained(model, device_map=dev,
                                                    dtype=torch.bfloat16)
        self.model = model

    @property
    def cache_key(self) -> str:
        return f"qwen-stress:{Path(self.reference).stem}:stress={self.stress}:v1"

    def mark(self, text: str) -> str:
        if self._acc is None:
            return text
        from ..ru_stress import to_acute_sparse
        words = text.split(" ")
        bare = " ".join(w.replace(ACUTE, "") for w in words)
        marked = self._acc.mark(bare).split(" ")
        if len(marked) == len(words):
            marked = [_acute_to_plus(w) if ACUTE in w else m for w, m in zip(words, marked)]
        return to_acute_sparse(" ".join(marked))

    def synth(self, text: str, *, lang: str = "ru", seed: int | None = None) -> Result:
        prompt = self.mark(text) if lang == "ru" else text
        ceiling = max(MIN_CEILING_S, len(text) * MAX_SECONDS_PER_CHAR)
        base = 0 if seed is None else seed
        t0 = time.perf_counter()
        for attempt in range(ATTEMPTS):
            self._torch.manual_seed(base + 1000 * attempt)
            wavs, sr = self._model.generate_voice_clone(
                text=prompt, language="Russian" if lang == "ru" else "Auto",
                ref_audio=self.reference, ref_text=self.reference_text)
            audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            if audio.size / sr <= ceiling:
                break
        self.sample_rate = sr
        return Result(audio=audio, sample_rate=sr, seconds=audio.size / sr,
                      synth_seconds=time.perf_counter() - t0)

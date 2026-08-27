"""Chatterbox Multilingual V3 backend (MIT, 23 languages incl. ru + en).

The reason to want this model: one 0.5B checkpoint covers both languages in the
library, clones a narrator voice from ~5s of reference audio, and so keeps a
single consistent voice across a mixed-language shelf.

Two operational facts measured on this project, both worth knowing before you
plan around it:

* **Environment.** chatterbox-tts pins torch 2.6, which cannot share a venv with
  the main project's torch. Run it in its own environment (or on the Spark,
  where it is the primary workload) rather than trying to unify them.
* **Speed.** On an M4 Max: RTF ~2.4-2.8 on CPU, and ~5.6 on MPS -- MPS is
  *slower*, not broken, so there is nothing to gain by enabling it here. At
  RTF 2.5 a 13.6-hour book is ~34 hours single-stream, which is why synthesis
  belongs on the GPU box.

Also note `perth` (the PerTh watermarker Chatterbox applies to every clip)
imports `pkg_resources`; with setuptools >= 81 it fails silently and surfaces
as `TypeError: 'NoneType' object is not callable` at model load. Pin
`setuptools<81`.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

# Chatterbox Multilingual V3 language ids relevant to this library.
SUPPORTED = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
}


class Chatterbox:
    name = "chatterbox"

    def __init__(
        self,
        reference: str | Path | None = None,
        device: str = "auto",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        stress: str = "ruaccent",
    ):
        from ..device import pick as pick_device
        # On Apple Silicon this model measured slower on MPS than on CPU, so a
        # bare "auto" would pick the worse option; CPU is chosen there.
        device = pick_device(device)
        if device == "mps":
            device = "cpu"
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.reference = str(reference) if reference else None
        self.stress = stress
        self._accent = None
        if stress == "ruaccent":
            from ..ru_stress import Accentuator
            self._accent = Accentuator()
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        self.sample_rate = self.model.sr

    def synth(self, text: str, *, lang: str = "ru", seed: int | None = None):
        from .base import Result
        import torch

        if lang not in SUPPORTED:
            raise ValueError(f"chatterbox has no language id {lang!r}")
        if seed is not None:
            torch.manual_seed(seed)

        if self._accent is not None and lang == "ru":
            from ..ru_stress import to_acute
            # '+' marks corrupt Chatterbox badly; the combining acute does not.
            text = to_acute(self._accent.mark(text))

        t0 = time.perf_counter()
        wav = self.model.generate(
            text, language_id=lang, audio_prompt_path=self.reference,
            exaggeration=self.exaggeration, cfg_weight=self.cfg_weight,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
        )
        dt = time.perf_counter() - t0
        arr = wav.squeeze().float().cpu().numpy().astype(np.float32)
        return Result(arr, self.sample_rate, arr.size / self.sample_rate, dt)

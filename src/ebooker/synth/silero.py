"""Silero v5 backend (Russian and other CIS languages).

Fixed speaker set, no voice cloning.

On stress: Silero has built-in accentuation (`put_accent`, `put_stress_homo`,
`put_yo_homo`) and it is *not* reliable enough for narration -- listening found
audibly wrong stress that the ASR round-trip cannot detect, because Whisper
transcribes за́мок and замо́к as the same string. Pass `stress="ruaccent"` to mark
the text with RUAccent first and set put_accent=False so Silero does not
overrule it. RUAccent resolves homographs from context ("Я из г+отов" the Goths
versus "гот+ов" ready) which Silero's own accentuation does not.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .base import Result

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "v5_5_ru.pt"
MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"

SPEAKERS = ["aidar", "baya", "kseniya", "eugene", "xenia"]


class Silero:
    name = "silero"

    def __init__(
        self,
        speaker: str = "eugene",
        sample_rate: int = 48000,
        model_path: str | Path = DEFAULT_MODEL,
        device: str = "cpu",
        threads: int | None = None,
        stress: str = "ruaccent",
    ):
        self.speaker = speaker
        self.stress = stress
        self._accent = None
        if stress == "ruaccent":
            from ..ru_stress import Accentuator
            self._accent = Accentuator()
        self.sample_rate = sample_rate
        self.device = device
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing -- download it with:\n  curl -L -o {path} {MODEL_URL}")
        # Silero is CPU-bound and scales with threads; MPS gives nothing here.
        if threads:
            torch.set_num_threads(threads)
        importer = torch.package.PackageImporter(str(path))
        self.model = importer.load_pickle("tts_models", "model")
        self.model.to(torch.device(device))

    def synth(self, text: str, *, lang: str = "ru", seed: int | None = None) -> Result:
        if seed is not None:
            torch.manual_seed(seed)
        marked = text
        external = self._accent is not None and lang == "ru"
        if external:
            marked = self._accent.mark(text)
        t0 = time.perf_counter()
        with torch.inference_mode():
            audio = self.model.apply_tts(
                text=marked,
                speaker=self.speaker,
                sample_rate=self.sample_rate,
                # When RUAccent has already marked the text, Silero must not
                # re-derive stress on top of it.
                put_accent=not external,
                put_stress_homo=not external,
                put_yo=not external,
                put_yo_homo=not external,
                stress_single_vowel=True,
            )
        dt = time.perf_counter() - t0
        arr = audio.detach().cpu().numpy().astype(np.float32)
        return Result(arr, self.sample_rate, len(arr) / self.sample_rate, dt)

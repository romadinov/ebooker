"""Shtorm-PocketTTS-RU backend (Russian, voice cloning).

A 100M-parameter Russian book-reading fine-tune of Kyutai's Pocket TTS:
`ArtShtorm/Shtorm-PocketTTS-RU` (CC-BY-4.0 weights, MIT code). It was picked by
ear over the stress-tuned Qwen clone, MOSS-TTS v1.5 (8B) and VoXtream2-RU on
the same 40 chunks of Б. Вавилонская, all cloning the same narrator from the
same 8.7 s clip -- and despite the lowest speaker-embedding match to his real
recordings of that round (0.61 against the Qwen clone's 0.89). The metric and
the ear disagree about this model; the ear decides.

Two checkpoints, chosen with ``--variant``: ``slow`` is the author's calmer
delivery and the one picked, ``fast`` the other. Both read at about 14
characters per second, the narrator's own pace.

Text treatment is the one qwen-stress uses, because the model takes the same
notation: ё restored and sparse combining-acute marks from silero (it was
trained on acute-marked text and follows the marks), mid-sentence dashes read
as an ellipsis. The author's advice: a reference of about 5 s ending on a
complete phrase, chunks of 120-180 characters (the pipeline's are 160: longer
ones lose volume towards the end), and no Latin letters or digits -- digits
the pipeline already spells out. No transcript of the reference is needed.

It is small enough to run on CPU, which is what Pocket TTS is for: 5 minutes of
audio rendered in 0.9 minutes on a DGX Spark's CPU cores, faster still on its
GPU. Mimi, the audio codec, stays on the CPU by the library's own design.

Guards: a chunk far longer than its text warrants is regenerated with another
seed, as for qwen-stress; so is one that comes out nearly empty, because an
immediate end-of-speech on short dialogue lines is exactly what an ill-suited
reference clip does to these models, and the runaway guard alone misses it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import Result
from .qwen_stress import MID_DASH, mark_acute

REPO = "ArtShtorm/Shtorm-PocketTTS-RU"
VARIANTS = ("slow", "fast")
MAX_SECONDS_PER_CHAR = 1 / 7.0   # it reads ~14 chars/s; twice that is a runaway
MIN_SECONDS_PER_CHAR = 1 / 40.0  # below this the chunk is silence, not speech
MIN_CEILING_S = 4.0
ATTEMPTS = 3


class Pocket:
    name = "pocket"
    sample_rate = 24000

    def __init__(
        self,
        reference: str | Path,
        device: str = "auto",
        stress: str = "silero",
        variant: str = "slow",
    ):
        import torch
        from pocket_tts import TTSModel

        from ..device import pick as pick_device

        if variant not in VARIANTS:
            raise ValueError(f"pocket variant must be one of {VARIANTS}, not {variant!r}")
        # oneDNN's JIT assembler aborts on some aarch64 CPUs (Xbyak: "label is
        # redefined"); the model is tiny, so the loss is nil.
        torch.backends.mkldnn.enabled = False
        self._torch = torch
        self.device = pick_device(device)
        self.reference = str(reference)
        self.variant = variant
        self.stress = stress
        self._acc = None
        if stress == "silero":
            from ..ru_stress import SileroAccentuator
            self._acc = SileroAccentuator()

        self._model = TTSModel.load_model(config=f"hf://{REPO}/config_{variant}.yaml")
        self._model.to(self.device)
        self.sample_rate = int(self._model.sample_rate)
        self._voice = self._model.get_state_for_audio_prompt(self.reference)

    @property
    def cache_key(self) -> str:
        return f"pocket:{Path(self.reference).stem}:{self.variant}:stress={self.stress}:v1"

    def mark(self, text: str) -> str:
        return mark_acute(self._acc, text)

    def synth(self, text: str, *, lang: str = "ru", seed: int | None = None) -> Result:
        prompt = self.mark(MID_DASH.sub("... ", text)) if lang == "ru" else text
        ceiling = max(MIN_CEILING_S, len(text) * MAX_SECONDS_PER_CHAR)
        floor = len(text) * MIN_SECONDS_PER_CHAR
        base = 0 if seed is None else seed
        t0 = time.perf_counter()
        for attempt in range(ATTEMPTS):
            self._torch.manual_seed(base + 1000 * attempt)
            y = self._model.generate_audio(self._voice, prompt)
            audio = np.asarray(y.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            seconds = audio.size / self.sample_rate
            if floor <= seconds <= ceiling:
                break
        return Result(audio=audio, sample_rate=self.sample_rate, seconds=seconds,
                      synth_seconds=time.perf_counter() - t0)

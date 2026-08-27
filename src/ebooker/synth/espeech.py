"""ESpeech-TTS-1 backend (Apache 2.0, F5/DiT, Russian).

Why this model: it is the only Russian option tested that combines
voice-cloning quality with *honoured* stress control. Chatterbox receives stress
marks and ignores them; Silero honours them but sounds flat; ESpeech's own
reference implementation is literally `if '+' in text: use as-is`, so RUAccent
marks reach the acoustic model intact.

Measured on an M4 Max via MPS, on the meteorite passage:
    ESpeech-RLV2   CER 0.7%   RTF 0.84
    ESpeech-SFT    CER 3.5%   RTF 0.78

**Word dropping is real and stochastic.** SFT rendered "шестнадцать" as "шить"
once; five different seeds on the same text all rendered it correctly. So the
ASR-verify-and-retry loop is not optional with this backend -- it is what makes
it usable. Note that a single mangled word inside a long chunk scores only ~5%
CER, under the 14% Russian threshold, which is why verify.py also checks numeral
and long-word integrity without length normalisation.

Environment: f5-tts needs its own venv. Also, torchaudio 2.11 routes load()
through torchcodec, whose dylib will not link against FFmpeg 9 -- audio I/O is
shimmed to soundfile below.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .base import Result

VARIANTS = {
    "rlv2": ("ESpeech/ESpeech-TTS-1_RL-V2", "espeech_tts_rlv2.pt"),
    "sft": ("ESpeech/ESpeech-TTS-1_SFT-256K", "espeech_tts_256k.pt"),
}
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

# ESpeech at speed 1.0 reads at ~21 chars/second against Silero's ~14, which is
# fast for narration. 0.82 lands near 18.
DEFAULT_SPEED = 0.82


def _shim_torchaudio() -> None:
    """Route torchaudio I/O through soundfile, bypassing torchcodec."""
    import soundfile as sf
    import torch
    import torchaudio

    if getattr(torchaudio, "_ebooker_shimmed", False):
        return

    def load(uri, *a, **kw):
        data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    def info(uri, *a, **kw):
        i = sf.info(str(uri))
        return type("Info", (), {"sample_rate": i.samplerate, "num_frames": i.frames,
                                 "num_channels": i.channels, "bits_per_sample": 16,
                                 "encoding": "PCM_S"})()

    torchaudio.load = load
    torchaudio.info = info
    torchaudio._ebooker_shimmed = True


class ESpeech:
    name = "espeech"
    sample_rate = 24000

    def __init__(
        self,
        reference: str | Path,
        reference_text: str | Path | None = None,
        variant: str = "rlv2",
        device: str = "auto",
        speed: float = DEFAULT_SPEED,
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        stress: str = "ruaccent",
    ):
        _shim_torchaudio()
        from ..device import pick as pick_device
        device = pick_device(device)
        from huggingface_hub import hf_hub_download
        from f5_tts.infer.utils_infer import (load_model, load_vocoder,
                                              preprocess_ref_audio_text)
        from f5_tts.model import DiT

        if variant not in VARIANTS:
            raise ValueError(f"unknown ESpeech variant {variant!r}; "
                             f"expected one of {sorted(VARIANTS)}")
        repo, fname = VARIANTS[variant]
        self.device = device
        self.speed = speed
        self.nfe_step = nfe_step
        self.cfg_strength = cfg_strength

        self._accent = None
        if stress == "ruaccent":
            from ..ru_stress import Accentuator
            self._accent = Accentuator()

        # Reference transcript: read from a sibling .txt if not supplied.
        if reference_text is None:
            cand = Path(reference).with_suffix(".txt")
            reference_text = cand.read_text(encoding="utf-8").strip() if cand.exists() else ""
        elif Path(str(reference_text)).exists():
            reference_text = Path(str(reference_text)).read_text(encoding="utf-8").strip()
        if self._accent is not None and reference_text and "+" not in reference_text:
            reference_text = self._accent.mark(reference_text)

        self.vocoder = load_vocoder(device=device)
        ckpt = hf_hub_download(repo_id=repo, filename=fname)
        vocab = hf_hub_download(repo_id=repo, filename="vocab.txt")
        self.model = load_model(DiT, MODEL_CFG, ckpt, vocab_file=vocab, device=device)
        self.ref_audio, self.ref_text = preprocess_ref_audio_text(
            str(reference), reference_text)
        self.batch_budget_bytes = self._budget()

    def _budget(self) -> int:
        """Bytes of text F5 will accept in a single batch for this reference.

        F5 sizes its batches from the reference:
            max_chars = ref_text_bytes / ref_secs * (22 - ref_secs) * speed
        Exceeding it makes infer_process split internally, and **that
        multi-batch path segfaults on MPS**. A longer reference therefore
        shrinks the budget: a 7.4 s clip allows ~309 bytes where a 3.6 s clip
        allows ~629. Since Cyrillic is two bytes per character, a 206-character
        chunk is ~390 bytes and overflows the longer reference.

        The backend splits to this budget itself, so infer_process is always
        called single-batch and the crash cannot occur.
        """
        import soundfile as sf

        info = sf.info(self.ref_audio) if isinstance(self.ref_audio, str) else None
        secs = (info.duration if info else 5.0)
        secs = min(max(secs, 0.5), 21.0)
        budget = int(len(self.ref_text.encode("utf-8")) / secs * (22 - secs) * self.speed)
        # Leave headroom: F5's own check is on the accumulated batch, and its
        # sentence splitter can overshoot slightly.
        return max(120, int(budget * 0.9))

    def _split_to_budget(self, text: str) -> list[str]:
        """Split at sentence boundaries so every piece fits one F5 batch."""
        budget = self.batch_budget_bytes
        if len(text.encode("utf-8")) <= budget:
            return [text]
        from ..normalise import split_sentences

        pieces, buf = [], ""
        for sent in split_sentences(text, "ru") or [text]:
            cand = (buf + " " + sent).strip() if buf else sent
            if buf and len(cand.encode("utf-8")) > budget:
                pieces.append(buf)
                buf = sent
            else:
                buf = cand
        if buf:
            pieces.append(buf)
        # A single sentence longer than the budget still has to be cut.
        out = []
        for p in pieces:
            while len(p.encode("utf-8")) > budget:
                cut = len(p) // 2
                sp = p.rfind(" ", 0, cut) or cut
                out.append(p[:sp].strip())
                p = p[sp:].strip()
            if p:
                out.append(p)
        return out

    def synth(self, text: str, *, lang: str = "ru", seed: int | None = None) -> Result:
        import torch
        from f5_tts.infer.utils_infer import infer_process

        if self._accent is not None and lang == "ru" and "+" not in text:
            text = self._accent.mark(text)
        if seed is not None:
            torch.manual_seed(seed)

        pieces = self._split_to_budget(text)
        t0 = time.perf_counter()
        chunks, sr = [], self.sample_rate
        for piece in pieces:
            wav, sr, _ = infer_process(
                self.ref_audio, self.ref_text, piece, self.model, self.vocoder,
                device=self.device, nfe_step=self.nfe_step,
                cfg_strength=self.cfg_strength, speed=self.speed,
                cross_fade_duration=0.15)
            chunks.append(np.asarray(wav, dtype=np.float32))
        dt = time.perf_counter() - t0
        arr = (chunks[0] if len(chunks) == 1
               else np.concatenate(chunks).astype(np.float32))
        self.sample_rate = sr

        # Apple's MPS layer JIT-compiles a Metal kernel per input shape, and
        # this model sees a different shape for almost every chunk. Over a long
        # book the kernel cache grows until specialising a new shape takes
        # minutes, which presents as a hang: the process stays alive at partial
        # CPU with no forward progress, and throughput degrades gradually
        # rather than stopping outright (measured: 10 min/chapter early,
        # ~31 min/chapter after 18 chapters). Draining the cache between chunks
        # keeps it bounded. This costs some recompilation but trades a hang for
        # steady-state slowness, which is the right way round for an
        # unattended overnight run.
        if self.device == "mps":
            try:
                import torch
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except Exception:
                pass
        return Result(arr, sr, arr.size / sr, dt)

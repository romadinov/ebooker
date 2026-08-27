"""Backend interface.

Everything downstream (verification, mastering, packaging) depends only on this,
so the Phase 0 bake-off winner can be swapped in without touching the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Result:
    audio: np.ndarray      # float32, mono, in [-1, 1]
    sample_rate: int
    seconds: float         # audio duration
    synth_seconds: float   # wall-clock spent generating

    @property
    def rtf(self) -> float:
        """Real-time factor: <1.0 is faster than real time."""
        return self.synth_seconds / self.seconds if self.seconds else float("inf")


class Backend(Protocol):
    name: str
    sample_rate: int

    def synth(self, text: str, *, lang: str, seed: int | None = None) -> Result: ...

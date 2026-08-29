"""Mastering tests, focused on the failure that cost two packaged books.

The vocoder emits an isolated NaN or Inf every few million samples. It is
inaudible, but the AAC encoder rejects the whole frame and ffmpeg aborts --
after rendering has already finished. These tests pin the guard that catches it.
"""

import numpy as np
import pytest

from ebooker import master


def test_sanitise_passes_clean_audio_through_unchanged():
    x = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    y, bad = master.sanitise(x)
    assert bad == 0
    assert np.array_equal(x, y)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_sanitise_replaces_each_kind_of_non_finite(value):
    x = np.zeros(1000, dtype=np.float32)
    x[500] = value
    y, bad = master.sanitise(x)
    assert bad == 1
    assert np.isfinite(y).all()
    assert y[500] == 0.0


def test_sanitise_counts_every_bad_sample():
    x = np.zeros(1000, dtype=np.float32)
    x[[10, 200, 999]] = [np.nan, np.inf, -np.inf]
    _, bad = master.sanitise(x)
    assert bad == 3


def test_sanitise_preserves_the_surrounding_signal():
    # One bad sample in a chapter must not disturb its neighbours: the repair
    # is a single 0.04 ms hole, not a rescaling of the waveform.
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.5, 0.5, 10_000).astype(np.float32)
    clean = x.copy()
    x[4321] = np.nan
    y, bad = master.sanitise(x)
    assert bad == 1
    mask = np.ones(x.size, dtype=bool)
    mask[4321] = False
    assert np.array_equal(y[mask], clean[mask])


def test_nan_survives_mastering_when_not_sanitised():
    # Documents why the guard sits before assemble(): a single NaN propagates
    # through the level statistics and contaminates the entire chapter.
    x = np.full(1000, 0.1, dtype=np.float32)
    x[0] = np.nan
    assert not np.isfinite(master.rms_dbfs(x))


def test_sanitised_audio_masters_cleanly():
    x = np.full(24_000, 0.1, dtype=np.float32)
    x[[5, 17_000]] = [np.nan, np.inf]
    y, bad = master.sanitise(x)
    assert bad == 2
    ch = master.assemble([(y, 0)], 24_000)
    assert np.isfinite(ch.audio).all()
    assert np.isfinite(ch.stats["peak_after_dbfs"])

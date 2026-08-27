"""Chunks -> a mastered chapter waveform.

Two responsibilities:

1. **Pacing.** Real silence between sentences and paragraphs. Without it the
   delivery is relentless, which is the most recognisable tell of a synthesised
   audiobook -- more so than timbre.
2. **Levels.** Hit the ACX window, which Apple Books accepts and which is the
   de-facto spoken-word standard: RMS between -23 and -18 dBFS, true peak at or
   below -3 dBFS, noise floor at or below -60 dBFS.

Measured on this corpus, ~5% of Silero chunks come out above full scale (peaks
to 1.25), so per-chunk headroom is applied before anything is summed -- if you
concatenate first and normalise after, those samples are already destroyed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TARGET_RMS_DBFS = -20.0     # centre of the ACX -23..-18 window
CEILING_DBFS = -3.0         # ACX true-peak limit
NOISE_FLOOR_DBFS = -60.0

TRIM_DB = -45.0             # below this counts as silence for trimming
TRIM_KEEP_MS = 40           # leave a little air so words are not clipped


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


def rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -np.inf
    return lin_to_db(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


# --------------------------------------------------------------------------

def headroom(x: np.ndarray, ceiling_db: float = CEILING_DBFS) -> np.ndarray:
    """Scale a chunk down if it exceeds the ceiling. Applied per chunk, before
    concatenation, because clipped samples cannot be recovered later."""
    peak = float(np.abs(x).max()) if x.size else 0.0
    limit = db_to_lin(ceiling_db)
    return x * (limit / peak) if peak > limit else x


def trim_silence(x: np.ndarray, sr: int, floor_db: float = TRIM_DB,
                 keep_ms: int = TRIM_KEEP_MS) -> np.ndarray:
    """Trim leading/trailing silence so inserted pauses are the only pauses.

    Models emit variable amounts of dead air at chunk edges; left in, it makes
    pacing unpredictable no matter what pause values are configured.
    """
    if x.size == 0:
        return x
    thresh = db_to_lin(floor_db)
    win = max(1, sr // 1000)                       # 1 ms envelope
    n = (x.size // win) * win
    if n == 0:
        return x
    env = np.abs(x[:n]).reshape(-1, win).max(axis=1)
    loud = np.flatnonzero(env > thresh)
    if loud.size == 0:
        return x[:0]
    keep = max(1, keep_ms)
    start = max(0, (loud[0] - keep)) * win
    end = min(x.size, (loud[-1] + 1 + keep) * win)
    return x[start:end]


def clamp_internal_silence(x: np.ndarray, sr: int, max_ms: int = 450,
                           floor_db: float = -42.0) -> tuple[np.ndarray, int]:
    """Shorten dead air *inside* a chunk to `max_ms`.

    Neural TTS sometimes inserts long internal pauses for reasons that are not
    in the text -- measured: ESpeech left 0.83 s around «Иоган Кеплер» purely
    because guillemets read as a dialogue boundary, and occasional 1.3 s gaps
    appear with no punctuation to explain them. Pacing is supposed to come from
    the pauses this module inserts deliberately, so anything longer inside a
    chunk is trimmed back rather than diagnosed case by case.

    Returns (audio, number of gaps shortened).
    """
    if x.size == 0:
        return x, 0
    win = max(1, sr // 100)                       # 10 ms resolution
    n = (x.size // win) * win
    if n == 0:
        return x, 0
    env = np.abs(x[:n]).reshape(-1, win).max(axis=1)
    quiet = env < db_to_lin(floor_db)
    keep_frames = max(1, int(max_ms / 10))

    out: list[np.ndarray] = []
    shortened = 0
    i = 0
    while i < quiet.size:
        if not quiet[i]:
            j = i
            while j < quiet.size and not quiet[j]:
                j += 1
            out.append(x[i * win:j * win])
            i = j
            continue
        j = i
        while j < quiet.size and quiet[j]:
            j += 1
        run = j - i
        if run > keep_frames:
            shortened += 1
            out.append(x[i * win:(i + keep_frames) * win])
        else:
            out.append(x[i * win:j * win])
        i = j
    if n < x.size:
        out.append(x[n:])
    return (np.concatenate(out).astype(np.float32) if out else x), shortened


def silence(ms: int, sr: int) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000), dtype=np.float32)


def soft_limit(x: np.ndarray, ceiling_db: float = CEILING_DBFS) -> np.ndarray:
    """tanh knee above the ceiling: holds the ACX peak limit without the
    audible edge of hard clipping."""
    limit = db_to_lin(ceiling_db)
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak <= limit:
        return x
    knee = limit * 0.7
    out = x.copy()
    over = np.abs(out) > knee
    if np.any(over):
        sign = np.sign(out[over])
        excess = (np.abs(out[over]) - knee) / max(limit - knee, 1e-9)
        out[over] = sign * (knee + (limit - knee) * np.tanh(excess))
    np.clip(out, -limit, limit, out=out)
    return out


def normalise(x: np.ndarray, target_db: float = TARGET_RMS_DBFS,
              ceiling_db: float = CEILING_DBFS) -> tuple[np.ndarray, dict]:
    """Gain to the RMS target, then limit peaks. Measures speech-only RMS so
    inter-sentence silence does not drag the average down and cause overshoot."""
    if x.size == 0:
        return x, {}
    speech = x[np.abs(x) > db_to_lin(-50.0)]
    before = rms_dbfs(speech if speech.size else x)
    gain = db_to_lin(target_db - before)
    y = soft_limit(x * gain, ceiling_db)
    after_speech = y[np.abs(y) > db_to_lin(-50.0)]
    return y, {
        "rms_before_dbfs": before,
        "gain_db": lin_to_db(gain),
        "rms_after_dbfs": rms_dbfs(after_speech if after_speech.size else y),
        "peak_after_dbfs": lin_to_db(float(np.abs(y).max())),
        # Every dB of gain lifts the noise floor by the same amount, so the
        # raw floor has to start low enough to survive it.
        "implied_noise_floor_dbfs": NOISE_FLOOR_DBFS - lin_to_db(gain),
    }


@dataclass
class Chapter:
    audio: np.ndarray
    sample_rate: int
    seconds: float
    stats: dict


def assemble(pieces: list[tuple[np.ndarray, int]], sr: int,
             lead_ms: int = 700, tail_ms: int = 900,
             max_internal_silence_ms: int = 450) -> Chapter:
    """Join (audio, pause_after_ms) pairs into one mastered chapter."""
    out: list[np.ndarray] = [silence(lead_ms, sr)]
    clamped = 0
    for audio, pause_ms in pieces:
        a = trim_silence(headroom(audio), sr)
        if max_internal_silence_ms:
            a, n = clamp_internal_silence(a, sr, max_internal_silence_ms)
            clamped += n
        if a.size:
            out.append(a)
            out.append(silence(pause_ms, sr))
    out.append(silence(tail_ms, sr))
    joined = np.concatenate(out).astype(np.float32)
    y, stats = normalise(joined)
    stats["internal_silences_clamped"] = clamped
    return Chapter(y, sr, y.size / sr, stats)

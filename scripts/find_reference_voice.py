"""Pick good voice-cloning reference clips out of a long audiobook.

A reference clip decides the timbre of an entire library, so it is worth
choosing rather than grabbing. F5/ESpeech wants a short (5-15 s), clean,
continuous stretch of ordinary narration, ending on a sentence boundary, with an
accurate transcript.

Candidates are scored on signal properties first (cheap), then the survivors are
transcribed and scored on how usable the text is. Things that make a clip bad:

* music or effects under the voice -- common at chapter openings;
* long internal silences, which F5 reproduces as dead air;
* clipping or wildly varying level;
* character voices rather than narration -- these carry over and make every
  book sound like that character;
* a transcript that ends mid-sentence, which teaches the model to trail off.

Usage:
    find_reference_voice.py <audiobook> <out_dir> [--n 8] [--seconds 11]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ebooker import verify as V
from ebooker.master import lin_to_db, rms_dbfs

SR = 24000


def probe_chapters(path: pathlib.Path) -> list[tuple[float, float]]:
    r = subprocess.run(["ffprobe", "-v", "error", "-of", "json",
                        "-show_chapters", "-show_format", str(path)],
                       capture_output=True, text=True)
    d = json.loads(r.stdout or "{}")
    chapters = [(float(c["start_time"]), float(c["end_time"]))
                for c in d.get("chapters", [])]
    if not chapters:
        dur = float(d.get("format", {}).get("duration", 0.0))
        chapters = [(0.0, dur)]
    return chapters


def decode(path: pathlib.Path, start: float, dur: float) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        wav = pathlib.Path(td) / "s.wav"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(path),
                        "-ac", "1", "-ar", str(SR), "-c:a", "pcm_f32le", str(wav)],
                       check=True)
        a, _ = sf.read(wav, dtype="float32")
    return a


def silence_fraction(a: np.ndarray, floor_db: float = -45.0) -> float:
    win = SR // 50                                  # 20 ms
    n = (a.size // win) * win
    if n == 0:
        return 1.0
    env = np.abs(a[:n]).reshape(-1, win).max(axis=1)
    return float((env < 10 ** (floor_db / 20)).mean())


def longest_gap_s(a: np.ndarray, floor_db: float = -45.0) -> float:
    win = SR // 50
    n = (a.size // win) * win
    if n == 0:
        return 0.0
    env = np.abs(a[:n]).reshape(-1, win).max(axis=1)
    quiet = env < 10 ** (floor_db / 20)
    best = cur = 0
    for q in quiet:
        cur = cur + 1 if q else 0
        best = max(best, cur)
    return best * win / SR


def tonality(a: np.ndarray) -> float:
    """Crude music detector: sustained narrowband energy scores high.

    Speech spectra move constantly; music under narration holds steady tones, so
    a high median of the per-frame spectral peak-to-mean ratio flags it.
    """
    win, hop = 2048, 1024
    if a.size < win * 4:
        return 0.0
    frames = [a[i:i + win] for i in range(0, a.size - win, hop)]
    ratios = []
    w = np.hanning(win)
    for f in frames[:400]:
        spec = np.abs(np.fft.rfft(f * w)) + 1e-9
        ratios.append(float(spec.max() / spec.mean()))
    return float(np.median(ratios))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audiobook")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=8, help="candidates to keep")
    ap.add_argument("--seconds", type=float, default=11.0)
    ap.add_argument("--stride", type=float, default=180.0,
                    help="seconds between sampled windows")
    ap.add_argument("--skip-intro", type=float, default=90.0,
                    help="seconds to skip at each chapter start (music/titles)")
    a = ap.parse_args()

    src = pathlib.Path(a.audiobook)
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chapters = probe_chapters(src)
    print(f"{src.name}: {len(chapters)} chapter(s)", flush=True)

    # Sample windows across the whole book, skipping chapter openings.
    windows: list[float] = []
    for st, en in chapters:
        t = st + a.skip_intro
        while t + a.seconds < en - 30:
            windows.append(t)
            t += a.stride
    print(f"sampling {len(windows)} windows of {a.seconds:.0f}s", flush=True)

    rows = []
    for i, t in enumerate(windows):
        try:
            seg = decode(src, t, a.seconds)
        except subprocess.CalledProcessError:
            continue
        if seg.size < SR * a.seconds * 0.9:
            continue
        peak = float(np.abs(seg).max())
        rows.append({
            "start": t,
            "rms": rms_dbfs(seg),
            "peak_db": lin_to_db(peak),
            "clipped": peak >= 0.999,
            "silence": silence_fraction(seg),
            "gap": longest_gap_s(seg),
            "tonality": tonality(seg),
        })
        if (i + 1) % 25 == 0:
            print(f"  scanned {i+1}/{len(windows)}", flush=True)

    # Signal-level shortlist.
    good = [r for r in rows
            if not r["clipped"]
            and -30 <= r["rms"] <= -12
            and r["silence"] < 0.30
            and r["gap"] < 0.60
            and r["tonality"] < 40]
    good.sort(key=lambda r: (r["silence"], r["tonality"]))
    print(f"{len(rows)} scanned, {len(good)} pass signal checks", flush=True)
    if not good:
        print("no clean windows found; loosen the thresholds", file=sys.stderr)
        return 1

    # Transcribe the best few and prefer clips that end on a sentence.
    tr = V.Transcriber()
    picks = []
    for r in good[: a.n * 3]:
        seg = decode(src, r["start"], a.seconds)
        text = tr(seg, SR, "ru").strip()
        words = len(text.split())
        if words < 8:
            continue
        # Trim to the last sentence-ending punctuation so the clip does not
        # teach the model to trail off mid-phrase.
        m = list(re.finditer(r"[.!?…]", text))
        complete = bool(m) and m[-1].end() >= len(text) - 2
        r["text"] = text
        r["words"] = words
        r["complete"] = complete
        picks.append(r)
        if len(picks) >= a.n * 2:
            break

    picks.sort(key=lambda r: (not r["complete"], r["silence"], -r["words"]))
    picks = picks[: a.n]

    from ebooker.ru_stress import Accentuator
    acc = Accentuator()
    manifest = []
    for i, r in enumerate(picks, 1):
        seg = decode(src, r["start"], a.seconds)
        stem = out / f"ref{i:02d}_{int(r['start'])}s"
        sf.write(stem.with_suffix(".wav"), seg, SR)
        marked = acc.mark(r["text"])
        stem.with_suffix(".txt").write_text(marked, encoding="utf-8")
        manifest.append({"file": stem.with_suffix(".wav").name, **r, "marked": marked})
        print(f"  {i:2}. {r['start']/60:7.1f} min  rms {r['rms']:6.1f} dB  "
              f"sil {r['silence']:.2f}  ton {r['tonality']:5.1f}  "
              f"{'complete' if r['complete'] else 'partial '}  {r['text'][:70]}",
              flush=True)

    (out / "candidates.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(picks)} candidates -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

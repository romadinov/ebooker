"""Turn a Russian single-speaker corpus into StyleTTS2 training lists.

The front-end here must be byte-for-byte the one used at inference, because the
whole point of this model is that stress is decided *before* the network sees
the text:

    text --silero--> ё restored, '+' stress --Russian-G2P rules--> IPA

See front.py for the front-end, and why it replaced espeak.

Output, in StyleTTS2's own format (`wav|phonemes|speaker`):

    <out>/wavs/*.wav      24 kHz mono PCM_16
    <out>/train_list.txt
    <out>/val_list.txt
    <out>/ood_texts.txt   phonemised sentences from *another* corpus, used by
                          the SLM adversarial stage as unseen text

Usage:
    prep.py css10  <corpus_dir> <out_dir> [--ood other_transcript]
    prep.py ruslan <corpus_dir> <out_dir> [--ood other_transcript]
"""
from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

SR = 24000
MIN_S, MAX_S = 1.0, 20.0
VAL_N = 100
from front import phonemise  # noqa: E402  (shared with infer.py)


def read_css10(root: Path) -> list[tuple[Path, str]]:
    rows = []
    for line in (root / "transcript.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            rows.append((root / parts[0], parts[1]))
    return rows


def read_ruslan(root: Path) -> list[tuple[Path, str]]:
    meta = next(root.rglob("metadata_RUSLAN_22200.csv"))
    wavs = {p.stem: p for p in root.rglob("*.wav")}
    rows = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        name, _, text = line.partition("|")
        if name in wavs:
            rows.append((wavs[name], text))
    return rows


READERS = {"css10": read_css10, "ruslan": read_ruslan}


def convert(args: tuple[Path, Path]) -> float | None:
    """Resample to 24 kHz mono; return duration, or None if out of range."""
    src, dst = args
    import librosa
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    dur = len(x) / sr
    if not MIN_S <= dur <= MAX_S or not np.isfinite(x).all():
        return None
    if not dst.exists():
        if sr != SR:
            x = librosa.resample(x, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        sf.write(tmp, x, SR, subtype="PCM_16", format="WAV")
        tmp.rename(dst)
    return dur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=READERS)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ood", type=Path, help="plain-text file, one sentence per line")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()

    rows = READERS[a.corpus](a.root)
    print(f"{len(rows)} utterances in {a.root}", flush=True)
    wav_dir = a.out / "wavs"
    names = [f"{a.corpus}_{i:05d}.wav" for i in range(len(rows))]

    with ProcessPoolExecutor(a.jobs) as ex:
        durs = list(ex.map(convert, [(r[0], wav_dir / n) for r, n in zip(rows, names)],
                           chunksize=64))
        keep = [i for i, d in enumerate(durs) if d is not None]
        print(f"kept {len(keep)} ({sum(durs[i] for i in keep) / 3600:.2f} h), "
              f"dropped {len(rows) - len(keep)} outside {MIN_S}-{MAX_S}s", flush=True)
        phones = list(ex.map(phonemise, [rows[i][1] for i in keep], chunksize=64))

    lines = [f"{names[i]}|{p}|0" for i, p in zip(keep, phones) if p]
    random.Random(0).shuffle(lines)
    (a.out / "val_list.txt").write_text("\n".join(lines[:VAL_N]) + "\n", encoding="utf-8")
    (a.out / "train_list.txt").write_text("\n".join(lines[VAL_N:]) + "\n", encoding="utf-8")

    if a.ood:
        texts = [t for t in a.ood.read_text(encoding="utf-8").splitlines() if t.strip()]
        with ProcessPoolExecutor(a.jobs) as ex:
            ood = list(ex.map(phonemise, texts, chunksize=64))
        (a.out / "ood_texts.txt").write_text(
            "\n".join(f"ood_{i}.wav|{p}|0" for i, p in enumerate(ood) if p) + "\n",
            encoding="utf-8")
        print(f"{len(ood)} OOD sentences", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

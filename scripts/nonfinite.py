#!/usr/bin/env python3
"""Find -- and optionally repair -- non-finite samples in rendered chapters.

Why this exists: the AAC encoder rejects any frame containing NaN or +-Inf and
aborts the whole encode, leaving behind the partial file it had already
written. The result looks like a success. Two finished books carried correct
chapter markers and plausible sizes while holding 0.27 h and 1.47 h of a
rendered 9.04 h and 11.73 h, and nothing in out/ said otherwise.

Measured incidence when it does occur: 66 samples across 52 of 240 chapters,
one to four per chapter, out of ~10 million samples each. A single sample is
0.04 ms and inaudible, so --repair replaces them with silence rather than
re-rendering, which would cost hours of GPU for a fault of that size.

ebooker.master.sanitise() now catches these at synthesis time, so new renders
should come back clean. This stays useful for auditing renders made before
that guard, and for confirming a book is encodable before committing to a long
packaging run.

    python scripts/nonfinite.py work/<book>/<cache-key>          # report
    python scripts/nonfinite.py work/<book>/<cache-key> --repair # fix in place

Exits non-zero if anything non-finite remains, so it can gate a build.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf


def scan(key: str) -> list[tuple[str, int, int, float]]:
    hits = []
    for path in sorted(glob.glob(os.path.join(key, "ch*.wav"))):
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        finite = np.isfinite(audio)
        bad = int(finite.size - int(finite.sum()))
        if bad:
            first = int(np.argmax(~finite)) / sr
            hits.append((path, bad, audio.size, first))
    return hits


def repair(path: str) -> None:
    info = sf.info(path)
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    sf.write(path, audio, sr, subtype=info.subtype)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("keys", nargs="+", help="cache directories holding ch*.wav")
    p.add_argument("--repair", action="store_true",
                   help="replace non-finite samples with silence, in place")
    a = p.parse_args(argv)

    total_bad = total_files = 0
    for key in a.keys:
        hits = scan(key)
        n = len(glob.glob(os.path.join(key, "ch*.wav")))
        print(f"\n=== {os.path.basename(os.path.dirname(key)) or key}  ({n} chapters)")
        if not hits:
            print("    clean")
        for path, bad, size, first in hits:
            total_bad += bad
            total_files += 1
            print(f"    {os.path.basename(path)}: {bad} non-finite of {size} "
                  f"samples ({100 * bad / size:.4f}%), first at {first:.1f}s")
            if a.repair:
                repair(path)

    if not total_bad:
        return 0
    print(f"\n{total_bad} non-finite sample(s) across {total_files} chapter(s)")
    if not a.repair:
        print("re-run with --repair to replace them with silence")
        return 1

    # Verify rather than trust: a partial repair still fails the encode, and
    # the whole point of this script is that the failure is otherwise silent.
    left = sum(bad for key in a.keys for _, bad, _, _ in scan(key))
    print(f"non-finite samples remaining after repair: {left}")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())

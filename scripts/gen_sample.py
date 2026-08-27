"""Render a demo passage THROUGH the production verify-and-retry path.

Earlier sample renders called the model directly and so bypassed verification.
That misrepresented the pipeline: the CLI flags and retries bad chunks, while the
demos shipped whatever came out first. Word drops that the pipeline would have
caught (ракетного, омнибуса, похожего) reached the listener.

Any script that produces audio to be listened to has to run the same checks the
real conversion does.

    gen_sample.py <label> <backend> <ref.wav> <variant> <speed> [--device mps]
"""
import argparse, pathlib, sys, time
import numpy as np, soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ebooker import master, verify as V

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--backend", default="espeech")
ap.add_argument("--ref", default="voices/ref_ru_short.wav")
ap.add_argument("--variant", default="sft")
ap.add_argument("--speed", type=float, default=0.90)
ap.add_argument("--device", default="mps")
ap.add_argument("--lang", default="ru")
ap.add_argument("--retries", type=int, default=3)
ap.add_argument("--passage", default="work/passages/passage_ru.txt")
a = ap.parse_args()

ROOT = pathlib.Path(__file__).resolve().parents[1]
if a.backend == "espeech":
    from ebooker.synth.espeech import _shim_torchaudio
    _shim_torchaudio()
from ebooker.synth import get_backend

texts = [l for l in (ROOT / a.passage).read_text("utf-8").split("\n") if l.strip()]
pauses = [350] * (len(texts) - 1) + [700]

kw = {}
if a.backend == "espeech":
    kw = {"reference": str(ROOT / a.ref), "variant": a.variant,
          "device": a.device, "speed": a.speed}
elif a.backend == "silero":
    kw = {"speaker": a.variant, "threads": 12}
elif a.backend == "kokoro":
    kw = {"lang_code": "b" if a.variant.startswith("bm_") else "a", "voice": a.variant}
tts = get_backend(a.backend, **kw)
tr = V.Transcriber()

pieces, flagged, retried = [], [], 0
t0 = time.perf_counter()
for i, text in enumerate(texts, 1):
    r = tts.synth(text, lang=a.lang, seed=1234)
    chk = V.check(r.audio, r.sample_rate, text, a.lang, transcriber=tr)
    attempts = [chk]
    for k in range(a.retries):
        if chk.ok:
            break
        retried += 1
        r = tts.synth(text, lang=a.lang, seed=1234 + 1000 * (k + 1))
        chk = V.check(r.audio, r.sample_rate, text, a.lang, transcriber=tr)
        attempts.append(chk)
    verdict = V.classify(attempts)
    if not chk.ok:
        flagged.append(f"chunk {i}: {verdict}: {'; '.join(chk.reasons)}")
    pieces.append((r.audio, pauses[i - 1]))
    print(f"  {i:2}/{len(texts)} {len(text):3}c {r.seconds:5.2f}s "
          f"{'ok' if chk.ok else verdict}", flush=True)

dt = time.perf_counter() - t0
m = master.assemble(pieces, tts.sample_rate)
out = ROOT / "work" / "passages" / f"passage__{a.label}.wav"
sf.write(out, m.audio, tts.sample_rate, subtype="FLOAT")
print(f"\nDONE {a.label}: {m.seconds:.1f}s audio, {dt:.1f}s wall, RTF {dt/m.seconds:.3f}")
print(f"  retries {retried}   flagged {len(flagged)}   "
      f"silences clamped {m.stats['internal_silences_clamped']}")
for f in flagged:
    print(f"  ! {f}")

"""Render a continuous passage per backend, for actual listening.

Short clips cannot answer "does this work as narration" -- pacing, breath and
consistency across sentence and paragraph boundaries only show up over tens of
seconds. This renders the same real passage through each backend, mastered
exactly as the delivered M4B would be, so the comparison is like-for-like.

Usage: gen_passage.py <epub> <lang> <out_dir> <seconds> [backend:voice ...]
"""
import pathlib, sys, time
import numpy as np, soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ebooker import ingest, master, normalise
from ebooker.synth import get_backend

epub, lang, outdir, target = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]), float(sys.argv[4])
specs = sys.argv[5:]
outdir.mkdir(parents=True, exist_ok=True)

book = ingest.load(epub)
chunks: list = []
for ch in book.chapters:
    got, _ = normalise.chunk_paragraphs(ch.paragraphs, lang, ch.index)
    chunks = got
    break
# Take whole chunks until the estimated duration reaches the target, so the
# passage always ends on a paragraph rather than mid-sentence.
picked, est = [], 0.0
for c in chunks:
    picked.append(c)
    est += 0.55 + c.char_count / 15
    if est >= target and c.is_paragraph_end:
        break
text_chars = sum(c.char_count for c in picked)
print(f"passage: {len(picked)} chunks, {text_chars} chars, ~{est:.0f}s est", flush=True)
(outdir / f"passage_{lang}.txt").write_text(
    "\n".join(c.text for c in picked), encoding="utf-8")

for spec in specs:
    parts = spec.split(":")
    name = parts[0]
    voice = parts[1] if len(parts) > 1 else ""
    extra = parts[2] if len(parts) > 2 else ""
    kw = {}
    if name == "silero":
        kw = {"speaker": voice or "eugene", "threads": 12}
        if extra:
            kw["stress"] = extra
    elif name == "kokoro":
        kw = {"lang_code": "b" if voice.startswith("bm_") else "a"}
        if voice:
            kw["voice"] = voice
    elif name == "chatterbox" and voice:
        kw = {"reference": voice}
    label = "-".join([p for p in (name, voice, extra) if p])
    try:
        tts = get_backend(name, **kw)
    except Exception as e:
        print(f"  ! {label}: {type(e).__name__}: {e}", flush=True)
        continue
    tts.synth("Warm up." if lang != "ru" else "Разогрев.", lang=lang)
    t0 = time.perf_counter()
    pieces = [(tts.synth(c.text, lang=lang).audio, c.pause_ms) for c in picked]
    dt = time.perf_counter() - t0
    mastered = master.assemble(pieces, tts.sample_rate)
    dest = outdir / f"passage__{label}.wav"
    sf.write(dest, mastered.audio, mastered.sample_rate, subtype="FLOAT")
    print(f"  {label:26} {mastered.seconds:5.1f}s  synth {dt:5.1f}s  "
          f"RTF {dt/mastered.seconds:.3f}  RMS {mastered.stats['rms_after_dbfs']:.1f} dBFS",
          flush=True)

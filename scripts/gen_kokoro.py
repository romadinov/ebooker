"""English bake-off: Kokoro-82M. Run with the kokoro venv.

Included because Kokoro is what every off-the-shelf EPUB-to-M4B tool
(audiblez, abogen, epub2tts-kokoro) is built on, so it is the de-facto English
baseline. Apache 2.0, 82M params, no voice cloning -- you get its built-in
voices.

Needs espeak-ng for out-of-vocabulary words. On macOS:
  brew install espeak-ng
  export ESPEAK_DATA_PATH=/opt/homebrew/opt/espeak-ng/share/espeak-ng-data
The bundled espeakng-loader ships a broken CI path, so setting this is required.
Its English G2P also needs the spaCy model en_core_web_sm.
"""
import json, pathlib, sys, time
import numpy as np, soundfile as sf

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "work/bakeoff_en")
OUT.mkdir(parents=True, exist_ok=True)
CASES = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "scripts/bakeoff_cases_en.json")
cases = json.loads(CASES.read_text("utf-8"))

from kokoro import KPipeline
SR = 24000
pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

rows = []
for voice in ("am_michael", "bm_george", "af_heart"):
    list(pipe("Warming up.", voice=voice))            # exclude warmup
    for c in cases:
        t0 = time.perf_counter()
        segs = list(pipe(c["text"], voice=voice))
        dt = time.perf_counter() - t0
        if not segs:
            print(f"  ! {voice} {c['id']}: no audio", flush=True)
            continue
        a = np.concatenate([s.output.audio.cpu().numpy() for s in segs]).astype(np.float32)
        secs = len(a) / SR
        name = f"kokoro-{voice}__{c['id']}.wav"
        sf.write(OUT / name, a, SR, subtype="FLOAT")
        rows.append({"backend": f"kokoro-{voice}", "case": c["id"], "file": name,
                     "text": c["text"], "lang": "en", "seconds": secs,
                     "synth_seconds": dt, "rtf": dt / secs, "sample_rate": SR})
        print(f"  kokoro-{voice:12} {c['id']:16} {secs:5.1f}s RTF={dt/secs:.3f}", flush=True)

(OUT / "gen_kokoro.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
print(f"\n{len(rows)} clips -> {OUT}")

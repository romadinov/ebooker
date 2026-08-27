"""English bake-off: the three Chatterbox variants. Run with the chatterbox venv.

English is where Chatterbox's blind-test record actually applies, and where it
has a dedicated English-only checkpoint plus a Turbo variant -- neither of which
exists for Russian. Worth testing all three rather than reusing the multilingual
number from the Russian run.
"""
import json, pathlib, sys, time
import numpy as np, soundfile as sf, torch

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "work/bakeoff_en")
OUT.mkdir(parents=True, exist_ok=True)
CASES = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "scripts/bakeoff_cases_en.json")
cases = json.loads(CASES.read_text("utf-8"))

def load_variants():
    out = []
    from chatterbox.tts import ChatterboxTTS
    out.append(("chatterbox-en-base", ChatterboxTTS.from_pretrained(device="cpu"), "en"))
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        out.append(("chatterbox-en-turbo",
                    ChatterboxTurboTTS.from_pretrained(device="cpu"), "en"))
    except Exception as e:
        print(f"  ! turbo unavailable: {type(e).__name__}: {e}", flush=True)
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    out.append(("chatterbox-mtl-en",
                ChatterboxMultilingualTTS.from_pretrained(device="cpu"), "mtl"))
    return out

rows = []
for label, model, kind in load_variants():
    for c in cases:
        torch.manual_seed(1234)
        t0 = time.perf_counter()
        if kind == "mtl":
            wav = model.generate(c["text"], language_id="en")
        else:
            wav = model.generate(c["text"])
        dt = time.perf_counter() - t0
        a = wav.squeeze().float().cpu().numpy()
        secs = len(a) / model.sr
        name = f"{label}__{c['id']}.wav"
        sf.write(OUT / name, a, model.sr, subtype="FLOAT")
        rows.append({"backend": label, "case": c["id"], "file": name,
                     "text": c["text"], "lang": "en", "seconds": secs,
                     "synth_seconds": dt, "rtf": dt / secs if secs else None,
                     "sample_rate": model.sr})
        print(f"  {label:22} {c['id']:16} {secs:5.1f}s RTF={dt/secs:.2f}", flush=True)
    del model

(OUT / "gen_chatterbox_en.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
print(f"\n{len(rows)} clips -> {OUT}")

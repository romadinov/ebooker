"""Bake-off generation: Chatterbox Multilingual. Run with the chatterbox venv.

Kept as a standalone script because chatterbox-tts pins torch 2.6 and cannot
share an environment with the main project.
"""
import json, pathlib, sys, time
import numpy as np, soundfile as sf, torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS as M

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "work/bakeoff")
OUT.mkdir(parents=True, exist_ok=True)
CASES = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("scripts/bakeoff_cases.json")
REF = sys.argv[3] if len(sys.argv) > 3 else None   # optional reference voice wav

cases = json.loads(CASES.read_text("utf-8"))
model = M.from_pretrained(device="cpu")            # MPS measured slower than CPU
variants = [("chatterbox-default", None, 0.5)]
if REF:
    variants.append((f"chatterbox-cloned", REF, 0.5))

rows = []
for label, ref, exag in variants:
    for c in cases:
        torch.manual_seed(1234)
        t0 = time.perf_counter()
        wav = model.generate(c["text"], language_id=c["lang"],
                             audio_prompt_path=ref, exaggeration=exag,
                             cfg_weight=0.5, temperature=0.8)
        dt = time.perf_counter() - t0
        a = wav.squeeze().float().cpu().numpy()
        secs = len(a) / model.sr
        name = f"{label}__{c['id']}.wav"
        sf.write(OUT / name, a, model.sr)
        rows.append({"backend": label, "case": c["id"], "file": name,
                     "text": c["text"], "lang": c["lang"], "seconds": secs,
                     "synth_seconds": dt, "rtf": dt / secs if secs else None,
                     "sample_rate": model.sr})
        print(f"  {label:20} {c['id']:20} {secs:5.1f}s RTF={dt/secs:.2f}", flush=True)

(OUT / "gen_chatterbox.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
print(f"\n{len(rows)} clips -> {OUT}")

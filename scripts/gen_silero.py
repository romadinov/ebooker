"""Bake-off generation: Silero v5 voices. Run with the project venv."""
import json, pathlib, sys, time
import soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ebooker.synth.silero import Silero

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "work/bakeoff")
OUT.mkdir(parents=True, exist_ok=True)
cases = json.loads((pathlib.Path(__file__).parent / "bakeoff_cases.json").read_text("utf-8"))

rows = []
for speaker in ("eugene", "aidar", "baya", "xenia"):
    b = Silero(speaker=speaker, threads=12)
    b.synth("Разогрев.")                      # exclude warmup from timings
    for c in cases:
        r = b.synth(c["text"], lang=c["lang"])
        name = f"silero-{speaker}__{c['id']}.wav"
        sf.write(OUT / name, r.audio, r.sample_rate)
        rows.append({"backend": f"silero-{speaker}", "case": c["id"], "file": name,
                     "text": c["text"], "lang": c["lang"], "seconds": r.seconds,
                     "synth_seconds": r.synth_seconds, "rtf": r.rtf,
                     "sample_rate": r.sample_rate})
        print(f"  {speaker:8} {c['id']:20} {r.seconds:5.1f}s RTF={r.rtf:.4f}", flush=True)

(OUT / "gen_silero.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
print(f"\n{len(rows)} clips -> {OUT}")

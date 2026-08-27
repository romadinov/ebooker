"""Score every bake-off clip objectively and emit a listening report.

Objective metrics only decide intelligibility (did the model say the words?)
and cost (how fast?). Aesthetics -- does it sound like a narrator -- is a human
judgement, which is what the generated HTML page is for.
"""
import html, json, pathlib, statistics as st, sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ebooker import verify as V

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "work/bakeoff")
# Hours of audio in the book these cases came from, for the full-book estimate.
BOOK_HOURS = float(sys.argv[2]) if len(sys.argv) > 2 else 13.6
rows = []
for f in sorted(OUT.glob("gen_*.json")):
    rows += json.loads(f.read_text("utf-8"))
if not rows:
    sys.exit(f"no gen_*.json in {OUT}")

tr = V.Transcriber()
print(f"scoring {len(rows)} clips with whisper-large-v3-turbo ...", flush=True)
for r in rows:
    a, sr = sf.read(OUT / r["file"], dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    r["peak"] = float(np.abs(a).max())
    r["rms_dbfs"] = 20 * np.log10(max(float(np.sqrt(np.mean(a**2))), 1e-9))
    chk = V.check(a, sr, r["text"], r["lang"], transcriber=tr)
    r["cer"] = chk.cer
    r["transcript"] = chk.transcript
    r["reasons"] = chk.reasons
    r["ok"] = chk.ok
    print(f"  {r['backend']:20} {r['case']:20} CER={chk.cer:6.1%} "
          f"peak={r['peak']:.2f} {'OK' if chk.ok else 'FLAG ' + '; '.join(chk.reasons)}",
          flush=True)

(OUT / "scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")

# ---- per-backend summary -------------------------------------------------
backends = sorted({r["backend"] for r in rows})
print(f"\n{'backend':22} {'med CER':>8} {'max CER':>8} {'med RTF':>9} "
      f"{'clip':>5} {'flags':>6}  full-book-est")
summary = {}
for b in backends:
    rs = [r for r in rows if r["backend"] == b]
    cers = [r["cer"] for r in rs if r["cer"] is not None]
    rtfs = [r["rtf"] for r in rs if r["rtf"]]
    med_rtf = st.median(rtfs)
    est_h = BOOK_HOURS * med_rtf
    summary[b] = {
        "median_cer": st.median(cers), "max_cer": max(cers),
        "median_rtf": med_rtf, "clipped": sum(1 for r in rs if r["peak"] >= 1.0),
        "flags": sum(1 for r in rs if not r["ok"]),
        "book_hours": est_h,
    }
    est = f"{est_h*60:.0f} min" if est_h < 1.5 else f"{est_h:.1f} h"
    print(f"{b:22} {st.median(cers):7.1%} {max(cers):7.1%} {med_rtf:9.4f} "
          f"{summary[b]['clipped']:5} {summary[b]['flags']:6}  {est:>12}")
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")

# ---- listening report ----------------------------------------------------
cases = [c for c in dict.fromkeys(r["case"] for r in rows)]
def cell(b, c):
    r = next((x for x in rows if x["backend"] == b and x["case"] == c), None)
    if not r:
        return "<td class=na>—</td>"
    cls = "bad" if not r["ok"] else ("warn" if r["cer"] and r["cer"] > 0.10 else "good")
    flags = f"<div class=flags>{html.escape('; '.join(r['reasons']))}</div>" if r["reasons"] else ""
    return (f"<td class={cls}><audio controls preload=none src='{html.escape(r['file'])}'></audio>"
            f"<div class=m>CER {r['cer']:.1%} · {r['seconds']:.1f}s · RTF {r['rtf']:.3g}"
            f" · peak {r['peak']:.2f}</div>{flags}"
            f"<details><summary>heard</summary><p>{html.escape(r['transcript'])}</p></details></td>")

parts = ["""<!doctype html><meta charset=utf-8><title>ebooker TTS bake-off</title>
<style>
body{font:15px/1.5 -apple-system,system-ui,sans-serif;margin:24px;max-width:100%}
h1{font-size:22px} h2{font-size:17px;margin-top:28px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #ccc;padding:6px 8px;vertical-align:top;text-align:left}
th{background:#f4f4f4;position:sticky;top:0}
td.good{background:#f2fbf3} td.warn{background:#fffbe9} td.bad{background:#fdf0f0}
.m{color:#555;font-size:11px;margin-top:4px;font-variant-numeric:tabular-nums}
.flags{color:#a33;font-size:11px;margin-top:3px}
.src{color:#333;background:#fafafa;border-left:3px solid #999;padding:6px 10px;margin:6px 0 10px}
audio{width:230px;height:32px} details{font-size:11px;margin-top:4px}
summary{cursor:pointer;color:#666} details p{margin:4px 0;color:#444}
.sum td,.sum th{font-variant-numeric:tabular-nums}
</style>
<h1>ebooker — TTS bake-off</h1>
<p>Objective columns decide <em>intelligibility</em> and <em>cost</em>. Which one sounds
like a narrator is yours to judge — play them side by side.
CER is measured by transcribing the generated audio with whisper-large-v3-turbo and
comparing to the input text, so proper nouns inflate it slightly for every backend equally.</p>
<h2>Summary</h2><table class=sum><tr><th>backend<th>median CER<th>max CER<th>median RTF
<th>clipped<th>flagged<th>est. full book</tr>"""]
for b in backends:
    s = summary[b]
    est = f"{s['book_hours']*60:.0f} min" if s['book_hours'] < 1.5 else f"{s['book_hours']:.1f} h"
    parts.append(f"<tr><td><b>{html.escape(b)}</b><td>{s['median_cer']:.1%}<td>{s['max_cer']:.1%}"
                 f"<td>{s['median_rtf']:.4g}<td>{s['clipped']}<td>{s['flags']}<td>{est}</tr>")
parts.append("</table>")

for c in cases:
    txt = next(r["text"] for r in rows if r["case"] == c)
    parts.append(f"<h2>{html.escape(c)}</h2><div class=src>{html.escape(txt)}</div>")
    parts.append("<table><tr>" + "".join(f"<th>{html.escape(b)}" for b in backends) + "</tr><tr>")
    parts.append("".join(cell(b, c) for b in backends))
    parts.append("</tr></table>")

(OUT / "report.html").write_text("\n".join(parts), "utf-8")
print(f"\nreport: {OUT/'report.html'}")

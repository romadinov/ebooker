"""A narrator's read-through of the text with Claude before synthesis.

The voice model reads exactly what it is given. Its stress follows the acute
marks, ё is read as ё, and every pause and inflection comes from punctuation;
it has no other controls. So the quality of a Russian chapter is set by the
marked text that reaches it, and the automatic marker is where the audible
faults came from: silero decides homographs from a narrow window and is wrong
about one time in ten on them (цвета́ for цве́та), and its ё dictionary has
gaps (раздёрнул). Pocket also gave a clause-joining comma no pause at all.

This pass hands each chapter, already marked, to ``claude -p`` in batches and
asks for a narrator's corrections: stress by grammar and context, ё, and
punctuation for the pauses and intonation that the model can actually render
(measured: comma ≈ 0, ellipsis ≈ 0.45 s, colon or dash ≈ 0.5 s, semicolon
≈ 0.55 s, full stop ≈ 0.7 s). Words may not change: a returned chunk whose
letters differ from the source, ignoring marks, punctuation and ё/е, is
rejected and the automatic marking stands. Decisions are cached per chapter
under ``work/<book>/edits/`` with a plain-text review file beside them, so a
re-render never asks twice and the edits can be read and overruled by hand.

Verification still compares the recording against the clean source text, so
marks and punctuation changes never count as errors there.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ACUTE = "́"
DEFAULT_MODEL = "opus"
BATCH = 20
JOBS = 4
PAUSE_LONG_EXTRA_MS = 600
PAUSE_SHORT_MIN_MS = 150

SYSTEM = """You prepare Russian prose for a text-to-speech narrator. The voice model reads exactly what it is given: it follows stress marks, reads ё as ё, and takes every pause and inflection from punctuation alone. It has no other controls.

You receive numbered chunks of a book already processed by an automatic tool: ё restored where it knew to, and a combining acute accent (U+0301) placed after the stressed vowel of most words of two or more syllables. The tool decides homographs from a narrow window of context and is wrong about one time in ten on them.

For each chunk, as the book's narrator would, do the following and nothing else:

1. Stress. Correct any mark that is wrong for the word's grammatical form in this sentence (цве́та "of the colour" vs цвета́ "colours"; за́мок vs замо́к; но́ги vs ноги́; бо́льшая vs больша́я; сто́ят vs стоя́т; проверочные формы of names and loanwords). Keep marks you agree with. Add a mark only to a word of two or more syllables that has none and whose reading could go wrong. Never mark a monosyllable or a word containing ё.
2. ё. Restore ё wherever the printed е stands for ё (ещё, раздёрнул, твёрдый, её, идёт, зелёный…). Leave е where it really is е (все vs всё by sense).
3. Pauses and intonation, through punctuation only. Measured on this voice: a comma often gets no pause at all; an ellipsis about 0.45 s; a colon or a dash about 0.5 s; a semicolon about 0.55 s; a full stop about 0.7 s. Where a comma joins two independent clauses, or precedes a clause the narrator would set off (…, надо…; …, значит…), use a semicolon or a dash. Where the author wants a held breath, an ellipsis. Make sure an exclamation ends with ! and a question with ?. Do not touch the dialogue dash at the start of a line. Inside «…» quotes change only marks and ё.
4. Never change, add, remove or reorder words or letters. The letters of your text, ignoring marks, punctuation, case and ё/е, must be exactly those of the input; a chunk whose letters differ is discarded.
5. "pause": "long" only when the chunk ends a scene or a weighty paragraph that wants a longer silence after it; "short" when it runs straight into the next chunk (an interrupted line, an enumeration). Omit otherwise.

Return only the chunks you changed, as JSON: {"chunks":[{"i":<number>,"text":"<corrected chunk>","pause":"long"}]} — "pause" is optional. If nothing needs changing return {"chunks":[]}. No commentary."""

SCHEMA = json.dumps({
    "type": "object",
    "properties": {"chunks": {"type": "array", "items": {
        "type": "object",
        "properties": {"i": {"type": "integer"}, "text": {"type": "string"},
                       "pause": {"type": "string", "enum": ["short", "long"]}},
        "required": ["i", "text"], "additionalProperties": False}}},
    "required": ["chunks"], "additionalProperties": False})


@dataclass
class Edit:
    text: str
    pause: str | None = None


def letters(s: str) -> str:
    """What may not change: the letters, with marks, case and ё/е collapsed."""
    s = unicodedata.normalize("NFD", s).replace(ACUTE, "")
    s = unicodedata.normalize("NFC", s).casefold().replace("ё", "е")
    return "".join(ch for ch in s if ch.isalnum())


def scale_pause(pause_ms: int, pause: str | None) -> int:
    if pause == "long":
        return pause_ms + PAUSE_LONG_EXTRA_MS
    if pause == "short":
        return max(PAUSE_SHORT_MIN_MS, pause_ms // 2)
    return pause_ms


def default_marker():
    """The pocket / qwen-stress text treatment without loading a voice model."""
    from .ru_stress import SileroAccentuator
    from .synth.qwen_stress import MID_DASH, mark_acute
    acc = SileroAccentuator()
    return lambda text: mark_acute(acc, MID_DASH.sub("... ", text))


def _claude(prompt: str, model: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    cmd = ["claude", "-p", "--output-format", "json", "--model", model,
           "--system-prompt", SYSTEM, "--json-schema", SCHEMA,
           "--tools", "", "--no-session-persistence"]
    res = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env,
                         timeout=900)
    if res.returncode != 0:
        raise RuntimeError(f"claude -p failed ({res.returncode}): {res.stderr[-400:]}")
    data = json.loads(res.stdout)
    out = data.get("structured_output")
    if out is None:
        text = data.get("result", "")
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        out = json.loads(text)
    return out


def edit_chapter(chunks, marker, out_dir: Path, chapter_index: int, *,
                 title: str = "", author: str = "", model: str = DEFAULT_MODEL,
                 batch: int = BATCH, jobs: int = JOBS, force: bool = False,
                 log=print) -> dict[int, Edit]:
    """Return {chunk index: Edit} for one chapter, from cache when present."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"ch{chapter_index:03d}.json"
    if cache.exists() and not force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        return {int(k): Edit(v["text"], v.get("pause")) for k, v in data["edits"].items()}

    by_index = {c.index: c for c in chunks}
    marked = {c.index: marker(c.text) for c in chunks}
    batches = [chunks[i:i + batch] for i in range(0, len(chunks), batch)]
    head = f"Chapter {chapter_index}" + (f" of «{title}»" if title else "") \
        + (f" by {author}" if author else "") + ".\n\n"

    def run(group):
        prompt = head + "\n".join(f"[{c.index}] {marked[c.index]}" for c in group)
        return _claude(prompt, model)

    edits: dict[int, Edit] = {}
    rejected: list[dict] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for group, fut in [(g, pool.submit(run, g)) for g in batches]:
            try:
                out = fut.result()
            except Exception as e:  # one failed batch keeps its automatic marks
                failed += 1
                log(f"    ch{chapter_index} editor batch {group[0].index}-{group[-1].index}: {e}")
                continue
            for item in out.get("chunks", []):
                i = item.get("i")
                text = (item.get("text") or "").strip()
                if i not in by_index or not text:
                    continue
                if letters(text) != letters(by_index[i].text):
                    rejected.append({"i": i, "text": text})
                    continue
                if text == marked[i] and not item.get("pause"):
                    continue
                edits[i] = Edit(text, item.get("pause"))

    cache.write_text(json.dumps({
        "model": model, "batch": batch, "chunks": len(chunks),
        "edits": {str(i): {"text": e.text, **({"pause": e.pause} if e.pause else {})}
                  for i, e in sorted(edits.items())},
        "rejected": rejected, "failed_batches": failed,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [f"# chapter {chapter_index}: {len(edits)} of {len(chunks)} chunks edited, "
             f"{len(rejected)} rejected, {failed} batch(es) failed ({model})"]
    for i, e in sorted(edits.items()):
        lines += [f"\n[{i}]" + (f" pause={e.pause}" if e.pause else ""),
                  f"  - {marked[i]}", f"  + {e.text}"]
    for r in rejected:
        lines += [f"\n[{r['i']}] REJECTED (letters changed)", f"  - {marked[r['i']]}", f"  + {r['text']}"]
    (out_dir / f"ch{chapter_index:03d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"    ch{chapter_index} editor: {len(edits)} edited, {len(rejected)} rejected"
        + (f", {failed} batch(es) failed" if failed else ""))
    return edits

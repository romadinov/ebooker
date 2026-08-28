"""ebooker CLI: inspect an EPUB, or convert one to an M4B audiobook."""

from __future__ import annotations

import argparse
import gc
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from . import ingest, master, normalise, package
from . import verify as verify_mod


def slugify(s: str, limit: int = 60) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s.-]", "", s, flags=re.UNICODE).strip()
    s = re.sub(r"[\s_]+", "-", s)
    return re.sub(r"-{2,}", "-", s).strip("-")[:limit] or "book"


def _current_rss_gb() -> float:
    """Resident size right now.

    Deliberately not resource.getrusage's ru_maxrss: that is a high-water mark
    which never falls, so it cannot distinguish memory being accumulated from
    memory merely peaking on a larger chapter.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:      # Linux
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    import resource                                                  # macOS
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3) if sys.platform == "darwin" else r / (1024 ** 2)


def parse_chapter_spec(spec: str, n: int) -> list[int]:
    """"21", "21-25", "1,3,7-9" -> sorted 1-based chapter numbers."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(i for i in out if 1 <= i <= n)


def cmd_inspect(a: argparse.Namespace) -> int:
    book = ingest.load(a.epub)
    print(f"title      : {book.title}")
    print(f"author     : {book.author}")
    print(f"language   : {book.language}")
    print(f"publisher  : {book.publisher} ({book.date})")
    print(f"cover      : {book.cover_path or '(none)'}")
    print(f"chapters   : {len(book.chapters)}")
    print(f"characters : {book.char_count:,}")
    print(f"est. audio : {book.char_count / 15 / 3600:.1f} h")
    if book.warnings:
        print("\nwarnings:")
        for w in book.warnings:
            print(f"  ! {w}")

    total_chunks, all_notes = 0, []
    for ch in book.chapters:
        chunks, notes = normalise.chunk_paragraphs(
            ch.paragraphs, book.language, ch.index, max_chars=a.max_chars)
        total_chunks += len(chunks)
        all_notes += notes
    print(f"\nchunks     : {total_chunks:,} (max {a.max_chars} chars)")
    print(f"norm. notes: {len(all_notes)}")
    if a.verbose:
        for n in all_notes[:40]:
            print(f"  - {n}")
        print("\nchapters:")
        for ch in book.chapters:
            print(f"  {ch.index:3} {ch.char_count:7,}  {ch.title}")
    return 0


def cmd_convert(a: argparse.Namespace) -> int:
    from .synth import default_backend, get_backend

    book = ingest.load(a.epub)
    lang = a.lang or book.language
    if lang in ("und", ""):
        print("error: no language in the EPUB; pass --lang", file=sys.stderr)
        return 2

    backend = a.backend or default_backend(lang)
    slug = slugify(book.title)
    # The cache key has to include the voice. Chapter wavs were previously kept
    # in one directory per book, so switching voice or backend silently reused
    # chapters rendered by the previous one -- a book with chapters 1-2 in an
    # American voice and 3-19 in a British one, with nothing to warn you.
    render_id = "-".join(str(x) for x in
                         (backend, a.voice or "default", a.variant if backend == "espeech" else "",
                          f"s{a.speed}" if a.speed else "", f"c{a.max_chars}")
                         if x)
    work = Path(a.work) / slug / slugify(render_id, limit=80)
    work.mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out) if a.out else Path("out") / f"{slug}.m4b"

    all_chapters = book.chapters
    if a.chapters:
        wanted = set(parse_chapter_spec(a.chapters, len(all_chapters)))
        chapters = [c for c in all_chapters if c.index in wanted]
        if not chapters:
            print(f"error: --chapters {a.chapters} selected nothing "
                  f"(book has {len(all_chapters)})", file=sys.stderr)
            return 2
    else:
        chapters = all_chapters[: a.limit] if a.limit else all_chapters
    print(f"{book.title} — {book.author}")
    print(f"{len(chapters)} chapter(s), {sum(c.char_count for c in chapters):,} chars, "
          f"lang={lang}, backend={backend}"
          + ("" if a.backend else " (auto-selected)"))
    from .device import describe as describe_device, mps_needs_restarts
    if backend in ("espeech", "chatterbox") and not a.package_only:
        print(f"device: {describe_device(a.device)}")
        if mps_needs_restarts(a.device) and not a.chapters:
            print("note: on MPS the Metal shape cache grows until the process "
                  "stalls; render a chapter at a time (--chapters N)")
    print(f"cache: {work}")

    # Packaging reads finished chapter wavs from the cache, so it must not
    # need a synthesis model -- requiring one meant you could not assemble a
    # book on a machine that lacked the backend that rendered it.
    tts = None
    transcriber = None
    kw: dict = {}
    if a.package_only:
        pass
    elif backend == "silero":
        kw = {"speaker": a.voice or "eugene", "threads": a.threads}
    elif backend == "kokoro":
        from .synth.kokoro import LANG_CODES
        kw = {"lang_code": LANG_CODES.get(lang, "a")}
        if a.voice:
            kw["voice"] = a.voice
    elif backend == "espeech":
        if not a.voice:
            print("error: --backend espeech needs --voice <reference.wav>",
                  file=sys.stderr)
            return 2
        kw = {"reference": a.voice, "variant": a.variant, "device": a.device}
        if a.speed:
            kw["speed"] = a.speed
    elif a.voice:
        kw = {"reference": a.voice}
    if not a.package_only:
        tts = get_backend(backend, **kw)

    if not (a.no_verify or a.package_only):
        transcriber = verify_mod.Transcriber()
    cover = ingest.extract_cover(a.epub, book, work)
    import resource

    specs: list[package.ChapterSpec] = []
    flagged: list[str] = []
    # Speaking rate differs per backend -- Silero runs near 14 chars/second,
    # ESpeech near 19.5 -- so a fixed prior over-predicts duration by ~30% on
    # the wrong model and rejects short lines that were fine. Calibrate from
    # what this backend actually produces.
    cps_samples: list[float] = []

    def current_cps() -> float:
        import statistics
        if len(cps_samples) < 12:
            return verify_mod.DEFAULT_CPS
        return statistics.median(cps_samples[-400:])
    t_start = time.perf_counter()
    audio_total = 0.0

    if a.package_only:
        chapters = []
    for ch in chapters:
        wav = work / f"ch{ch.index:03d}.wav"
        chunks, _ = normalise.chunk_paragraphs(
            ch.paragraphs, lang, ch.index, max_chars=a.max_chars)
        if wav.exists() and not a.force:
            info = sf.info(wav)
            specs.append(package.ChapterSpec(ch.title, wav, info.duration))
            audio_total += info.duration
            print(f"  ch{ch.index:3} cached  {info.duration/60:5.1f} min")
            continue

        t0 = time.perf_counter()
        pieces: list[tuple[np.ndarray, int]] = []
        chapter_flags: list[str] = []
        bad = 0
        retried = 0
        for c in chunks:
            r = tts.synth(c.text, lang=lang, seed=a.seed)
            if transcriber is not None:
                for attempt in range(a.retries + 1):
                    chk = verify_mod.check(r.audio, r.sample_rate, c.text, lang,
                                           transcriber=transcriber,
                                           cps=current_cps())
                    if chk.ok:
                        break
                    if attempt < a.retries:
                        retried += 1
                        r = tts.synth(c.text, lang=lang,
                                      seed=(a.seed or 0) + 1000 * (attempt + 1))
                else:
                    bad += 1
                    line = (f"ch{ch.index} chunk{c.index}: "
                            f"{'; '.join(chk.reasons)} :: {c.text[:60]}")
                    flagged.append(line)
                    chapter_flags.append(line)
            if r.seconds > 0.05 and len(c.text) >= 40:
                # Only well-sized chunks inform the rate estimate; short ones
                # are dominated by onset and trailing silence.
                cps_samples.append(len(c.text) / r.seconds)
            pieces.append((r.audio, c.pause_ms))

        mastered = master.assemble(pieces, tts.sample_rate)
        # float WAV: PCM_16 would clamp over-unity peaks that mastering repairs.
        sf.write(wav, mastered.audio, mastered.sample_rate, subtype="FLOAT")
        specs.append(package.ChapterSpec(ch.title, wav, mastered.seconds))
        audio_total += mastered.seconds
        rss_gb = _current_rss_gb()
        # Append this chapter's review lines now. Holding them until the run
        # ends loses everything if it is interrupted, and makes a long run
        # impossible to inspect while it is going.
        if chapter_flags:
            with (work / "flagged.txt").open("a", encoding="utf-8") as fh:
                fh.write("\n".join(chapter_flags) + "\n")

        # Release the chapter's audio before starting the next one. Measured on
        # a 10-chapter book in a single process: RSS climbed 15.3 -> 22.2 GB with
        # current RSS within 2% of the peak, so nothing was being returned. A
        # 128-chapter book in one process is an out-of-memory risk on that
        # trajectory, so drop the references and collect explicitly.
        pieces.clear()
        del mastered
        gc.collect()
        if backend in ("espeech", "chatterbox"):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        print(f"  ch{ch.index:3} {len(chunks):4} chunks  {mastered.seconds/60:5.1f} min  "
              f"{time.perf_counter()-t0:6.1f}s  rss {rss_gb:4.1f}G  "
              f"RMS {mastered.stats['rms_after_dbfs']:5.1f} "
              f"peak {mastered.stats['peak_after_dbfs']:5.1f} dBFS"
              + (f"  [{retried} retries]" if retried else "")
              + (f"  [{bad} flagged]" if bad else ""))

    elapsed = time.perf_counter() - t_start
    if a.render_only:
        rendered = len(list(work.glob("ch*.wav")))
        print(f"\nrendered {len(specs)} chapter(s) in {elapsed/60:.1f} min; "
              f"{rendered}/{len(all_chapters)} cached in {work}")
        if flagged:
            print(f"{len(flagged)} chunk(s) need review -> {work / 'flagged.txt'}")
        return 0

    # Package from the cache so a chapter-at-a-time run can assemble at the end.
    specs = []
    missing = []
    for ch in all_chapters:
        wav = work / f"ch{ch.index:03d}.wav"
        if wav.exists():
            specs.append(package.ChapterSpec(ch.title, wav, sf.info(wav).duration))
        else:
            missing.append(ch.index)
    if missing:
        print(f"error: {len(missing)} chapter(s) not rendered yet: "
              f"{missing[:12]}{'...' if len(missing) > 12 else ''}", file=sys.stderr)
        return 2
    audio_total = sum(s.seconds for s in specs)

    meta = {"title": book.title, "artist": book.author, "album": book.title,
            "album_artist": book.author, "date": book.date, "genre": "Audiobook",
            "comment": f"{backend}/{a.voice or 'default'}; publisher: {book.publisher}"}
    package.build(specs, meta, out_path, cover=cover, work_dir=work,
                  bitrate=a.bitrate)
    ok, problems = package.verify_m4b(out_path)

    print(f"\n{out_path}  {out_path.stat().st_size/1e6:.1f} MB  "
          f"{audio_total/3600:.2f} h audio")
    print(f"wall clock {elapsed/60:.1f} min  (RTF {elapsed/audio_total:.4f})")
    print(f"m4b checks: {'PASS' if ok else 'FAIL — ' + '; '.join(problems)}")
    if flagged:
        print(f"{len(flagged)} chunk(s) need review -> {work / 'flagged.txt'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ebooker",
                                description="EPUB -> Apple audiobook (M4B)")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("inspect", help="parse an EPUB and report structure")
    i.add_argument("epub")
    i.add_argument("--max-chars", type=int, default=normalise.MAX_CHUNK_CHARS)
    i.add_argument("-v", "--verbose", action="store_true")
    i.set_defaults(fn=cmd_inspect)

    c = sub.add_parser("convert", help="convert an EPUB to an M4B")
    c.add_argument("epub")
    c.add_argument("--backend", default=None,
                   choices=("silero", "kokoro", "espeech", "chatterbox"),
                   help="default: chosen per language (ru->silero, en->kokoro)")
    c.add_argument("--variant", default="rlv2", choices=("rlv2", "sft"),
                   help="espeech only: which checkpoint")
    c.add_argument("--device", default="auto",
                   help="cuda | mps | cpu | auto (default: detect)")
    c.add_argument("--speed", type=float, default=None,
                   help="espeech only: 1.0 is ~21 chars/s, 0.82 is ~18")
    c.add_argument("--voice", default=None,
                   help="silero speaker / kokoro voice, or reference wav for chatterbox")
    c.add_argument("--lang", default=None, help="override the EPUB's dc:language")
    c.add_argument("--out", default=None)
    c.add_argument("--work", default="work")
    c.add_argument("--limit", type=int, default=0, help="first N chapters only")
    c.add_argument("--chapters", default=None,
                   help="render only these chapters: \"21\", \"21-25\", \"1,3,7-9\"")
    c.add_argument("--render-only", action="store_true",
                   help="synthesise into the cache and stop, without packaging")
    c.add_argument("--package-only", action="store_true",
                   help="package the cached chapters, synthesising nothing")
    c.add_argument("--max-chars", type=int, default=normalise.MAX_CHUNK_CHARS)
    c.add_argument("--bitrate", default=package.BITRATE)
    c.add_argument("--threads", type=int, default=0)
    c.add_argument("--seed", type=int, default=None)
    c.add_argument("--retries", type=int, default=2)
    c.add_argument("--no-verify", action="store_true",
                   help="skip ASR round-trip QC (faster, riskier)")
    c.add_argument("--force", action="store_true", help="re-render cached chapters")
    c.set_defaults(fn=cmd_convert)

    a = p.parse_args(argv)
    return a.fn(a)

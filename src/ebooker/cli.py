"""ebooker CLI: inspect an EPUB, or convert one to an M4B audiobook."""

from __future__ import annotations

import argparse
import gc
import hashlib
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

from . import ingest, llm_edit, master, normalise, package
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
            ch.paragraphs, book.language, ch.index, max_chars=a.max_chars,
            latin=a.latin)
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


def cmd_edit(a: argparse.Namespace) -> int:
    """Run Claude's read-through on its own, so the edits can be reviewed (or
    produced on a machine that has `claude`) before a render elsewhere."""
    book = ingest.load(a.epub)
    lang = a.lang or book.language
    if (lang or "").split("-")[0] != "ru":
        print("error: the editor is for Russian text", file=sys.stderr)
        return 2
    chapters = book.chapters
    if a.chapters:
        wanted = set(parse_chapter_spec(a.chapters, len(chapters)))
        chapters = [c for c in chapters if c.index in wanted]
    edits_dir = Path(a.work) / slugify(book.title) / "edits"
    marker = llm_edit.default_marker()
    print(f"{book.title} — {book.author}: {len(chapters)} chapter(s), "
          f"model {a.editor_model}, {a.editor_jobs} jobs -> {edits_dir}")
    total = 0
    for ch in chapters:
        chunks, _ = normalise.chunk_paragraphs(
            ch.paragraphs, lang, ch.index, max_chars=a.max_chars, latin=a.latin)
        t0 = time.perf_counter()
        edits = llm_edit.edit_chapter(
            [c for c in chunks if not c.lang], marker, edits_dir, ch.index,
            title=book.title, author=book.author,
            model=a.editor_model, jobs=a.editor_jobs, force=a.force)
        total += len(edits)
        print(f"  ch{ch.index:3} {len(chunks):4} chunks  {len(edits):4} edited  "
              f"{time.perf_counter()-t0:5.0f}s  review: {edits_dir}/ch{ch.index:03d}.txt",
              flush=True)
    print(f"{total} chunks edited")
    return 0


def cmd_convert(a: argparse.Namespace) -> int:
    from .synth import default_backend, get_backend

    book = ingest.load(a.epub)
    lang = a.lang or book.language
    if lang in ("und", ""):
        print("error: no language in the EPUB; pass --lang", file=sys.stderr)
        return 2

    backend = a.backend or default_backend(lang)
    if backend == "pocket" and a.variant not in ("slow", "fast"):
        a.variant = "slow"   # --variant's default names an ESpeech checkpoint
    slug = slugify(book.title)
    # The cache key has to include the voice. Chapter wavs were previously kept
    # in one directory per book, so switching voice or backend silently reused
    # chapters rendered by the previous one -- a book with chapters 1-2 in an
    # American voice and 3-19 in a British one, with nothing to warn you.
    render_id = "-".join(str(x) for x in
                         (backend, a.voice or "default",
                          a.variant if backend in ("espeech", "pocket") else "",
                          # Stress marker changes the audio, so it must change
                          # the key: silero restores ё where RUAccent does not.
                          a.stress if (lang == "ru" and a.stress != "ruaccent") else "",
                          f"s{a.speed}" if a.speed else "", f"c{a.max_chars}",
                          # Claude's edits change the text the model reads.
                          "claude" if a.editor == "claude" else "")
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
    elif backend == "omnivoice":
        if not a.voice:
            print("error: --backend omnivoice needs --voice <reference.wav>",
                  file=sys.stderr)
            return 2
        kw = {"reference": a.voice, "device": a.device}
        if a.speed:
            kw["speed"] = a.speed
    elif backend == "piper":
        kw = {"voice": a.voice, "stress": "none" if a.stress == "none" else "silero"}
        if a.speed:
            kw["speed"] = a.speed
    elif backend == "qwen-stress":
        if not a.voice:
            print("error: --backend qwen-stress needs --voice <reference.wav> "
                  "(its transcript in <reference>.txt)", file=sys.stderr)
            return 2
        kw = {"reference": a.voice, "device": a.device,
              "stress": "none" if a.stress == "none" else "silero"}
    elif backend == "pocket":
        if not a.voice:
            print("error: --backend pocket needs --voice <reference.wav>",
                  file=sys.stderr)
            return 2
        kw = {"reference": a.voice, "device": a.device, "variant": a.variant,
              "stress": "none" if a.stress == "none" else "silero"}
    elif backend == "higgs":
        if not a.voice:
            print("error: --backend higgs needs --voice <reference.wav>",
                  file=sys.stderr)
            return 2
        kw = {"reference": a.voice, "device": a.device}
    elif backend == "espeech":
        if not a.voice:
            print("error: --backend espeech needs --voice <reference.wav>",
                  file=sys.stderr)
            return 2
        kw = {"reference": a.voice, "variant": a.variant, "device": a.device,
              "stress": a.stress}
        if a.speed:
            kw["speed"] = a.speed
    elif a.voice:
        kw = {"reference": a.voice}
    # Sentences in Latin script inside a Russian book go to an English voice
    # (normalise.Chunk.lang == "en"). Cloning backends read English from the
    # same clip; Pocket's Russian fine-tune cannot, so it gets Kyutai's English
    # model with the same clip (or a preset voice). Decided up front, so a
    # missing English voice fails before an hour of rendering, not during it.
    tts_en = None
    needs_en = False
    if lang == "ru" and a.latin == "route" and not a.package_only:
        for ch in chapters:
            cks, _ = normalise.chunk_paragraphs(ch.paragraphs, lang, ch.index,
                                                max_chars=a.max_chars, latin=a.latin)
            if any(c.lang == "en" for c in cks):
                needs_en = True
                break
    if not a.package_only:
        tts = get_backend(backend, **kw)
        if a.editor == "claude" and not hasattr(tts, "mark"):
            print(f"error: --editor claude needs a backend that reads acute stress "
                  f"marks (pocket, qwen-stress), not {backend}", file=sys.stderr)
            return 2
        if needs_en:
            if backend == "pocket" and a.en_backend == "qwen-stress":
                # The stress-tuned Qwen clone reads English from the same clip in
                # the narrator's own voice; picked by ear over Kyutai's presets.
                tts_en = get_backend("qwen-stress", reference=a.en_voice or a.voice,
                                     device=a.device, stress="none")
                print(f"English sentences: qwen-stress cloning {Path(a.en_voice or a.voice).stem}")
            elif backend == "pocket":
                en_voice = a.en_voice or a.voice
                try:
                    tts_en = get_backend("pocket", reference=en_voice, device=a.device,
                                         variant="english", stress="none")
                except RuntimeError as e:
                    print(f"error: {e}", file=sys.stderr)
                    print("       (or pass --latin translit to have the Russian voice "
                          "read the English sentences in Cyrillic)", file=sys.stderr)
                    return 2
                who = en_voice if Path(en_voice).suffix == "" else Path(en_voice).stem
                print(f"English sentences: Kyutai Pocket TTS, voice {who}")
            elif backend in ("qwen-stress", "higgs", "omnivoice", "espeech", "chatterbox"):
                tts_en = tts   # cloning models read English from the same clip
                print("English sentences: the same voice, unmarked text")
            else:
                print(f"error: {backend} cannot read the English sentences this book "
                      f"contains; pass --latin translit (Cyrillic transliteration, read "
                      f"by the Russian voice) or --latin keep", file=sys.stderr)
                return 2
            if tts_en.sample_rate != tts.sample_rate:
                print(f"error: English voice runs at {tts_en.sample_rate} Hz, the book's "
                      f"at {tts.sample_rate} Hz", file=sys.stderr)
                return 2
    edits_dir = Path(a.work) / slug / "edits"

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
            ch.paragraphs, lang, ch.index, max_chars=a.max_chars, latin=a.latin)
        if wav.exists() and not a.force:
            info = sf.info(wav)
            specs.append(package.ChapterSpec(ch.title, wav, info.duration))
            audio_total += info.duration
            print(f"  ch{ch.index:3} cached  {info.duration/60:5.1f} min")
            continue

        t0 = time.perf_counter()
        edits: dict[int, llm_edit.Edit] = {}
        if a.editor == "claude":
            edits = llm_edit.edit_chapter(
                [c for c in chunks if not c.lang], tts.mark, edits_dir, ch.index,
                title=book.title, author=book.author, model=a.editor_model,
                jobs=a.editor_jobs)
        pieces: list[tuple[np.ndarray, int]] = []
        chapter_flags: list[str] = []
        bad = 0
        retried = 0
        nonfinite = 0
        # Chunk-level cache. Chapters are the natural unit at 0.1 RTF, where a
        # chapter is minutes of work. At 3.5 RTF one chapter of this corpus is
        # three hours, so an interruption forfeits all of it -- measured: 2h20m
        # lost to a single stop 62% through a 448-chunk chapter. Keyed on a hash
        # of the chunk text, so edited text invalidates itself rather than
        # silently serving stale audio. Removed once the chapter wav exists.
        chunk_dir = work / f"chunks-ch{ch.index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        reused = 0

        # The chunk key must cover the synthesis settings, not just the text.
        # Keyed on text alone, changing temperature would silently reuse audio
        # generated under the old setting -- which is exactly the mistake that
        # would have kept 7.7% runaway chunks after fixing the cause.
        synth_keys = {
            "": str(getattr(tts, "cache_key", getattr(tts, "name", ""))),
            "en": str(getattr(tts_en, "cache_key", getattr(tts_en, "name", ""))),
        }
        for c in chunks:
            e = edits.get(c.index)
            # What the model reads; verification still compares with c.text.
            stext = e.text if e else c.text
            pause_ms = llm_edit.scale_pause(c.pause_ms, e.pause if e else None)
            # An English sentence goes to the English voice and is checked as English.
            ctts = tts_en if c.lang == "en" and tts_en is not None else tts
            clang = c.lang or lang
            if c.lang == "en":
                stext = stext.replace("«", '"').replace("»", '"')
            digest = hashlib.sha1(
                (synth_keys.get(c.lang, synth_keys[""]) + "\x00" + stext).encode("utf-8")).hexdigest()[:12]
            cpath = chunk_dir / f"c{c.index:04d}-{digest}.wav"
            if cpath.exists():
                try:
                    cached, csr = sf.read(cpath, dtype="float32", always_2d=False)
                    if cached.size and np.isfinite(cached).all():
                        pieces.append((cached, pause_ms))
                        reused += 1
                        continue
                except Exception:
                    cpath.unlink(missing_ok=True)   # truncated by a hard kill

            r = ctts.synth(stext, lang=clang, seed=a.seed)
            r.audio, n_bad = master.sanitise(r.audio)
            nonfinite += n_bad
            if transcriber is not None:
                for attempt in range(a.retries + 1):
                    chk = verify_mod.check(r.audio, r.sample_rate, c.text, clang,
                                           transcriber=transcriber,
                                           cps=current_cps())
                    if chk.ok:
                        break
                    if attempt < a.retries:
                        retried += 1
                        r = ctts.synth(stext, lang=clang,
                                       seed=(a.seed or 0) + 1000 * (attempt + 1))
                else:
                    bad += 1
                    line = (f"ch{ch.index} chunk{c.index}: "
                            f"{'; '.join(chk.reasons)} :: {c.text[:60]}")
                    flagged.append(line)
                    chapter_flags.append(line)
            # Heartbeat. Chapter-granularity logging is fine at 0.1 RTF but
            # leaves a slow backend silent for half an hour, during which a
            # stalled run and a working one look identical.
            if c.index and c.index % 10 == 0:
                done = sum(p[0].size for p in pieces) / tts.sample_rate
                el = time.perf_counter() - t0
                print(f"    ch{ch.index} chunk {c.index}/{len(chunks)}  "
                      f"{done/60:.1f} min audio  {el/60:.1f} min elapsed  "
                      f"rtf {el/max(done, 1e-6):.2f}", flush=True)
            if r.seconds > 0.05 and len(c.text) >= 40 and not c.lang:
                # Only well-sized chunks inform the rate estimate; short ones
                # are dominated by onset and trailing silence.
                cps_samples.append(len(c.text) / r.seconds)
            # format is explicit: soundfile infers it from the extension, and
            # the ".part" suffix (needed so a kill mid-write cannot leave a
            # half-file that reads as valid) carries no format information.
            tmp = cpath.with_suffix(".part")
            sf.write(tmp, r.audio, r.sample_rate, subtype="FLOAT", format="WAV")
            tmp.replace(cpath)
            pieces.append((r.audio, pause_ms))

        if reused:
            print(f"    ch{ch.index} reused {reused}/{len(chunks)} cached chunks",
                  flush=True)

        if nonfinite:
            line = (f"ch{ch.index}: {nonfinite} non-finite sample(s) "
                    f"replaced with silence")
            flagged.append(line)
            chapter_flags.append(line)

        mastered = master.assemble(pieces, tts.sample_rate)
        # float WAV: PCM_16 would clamp over-unity peaks that mastering repairs.
        sf.write(wav, mastered.audio, mastered.sample_rate, subtype="FLOAT")
        for stale in chunk_dir.glob("c*.wav"):
            stale.unlink(missing_ok=True)
        chunk_dir.rmdir() if not any(chunk_dir.iterdir()) else None
        specs.append(package.ChapterSpec(ch.title, wav, mastered.seconds))
        audio_total += mastered.seconds
        rss_gb = _current_rss_gb()
        # Append this chapter's review lines now. Holding them until the run
        # ends loses everything if it is interrupted, and makes a long run
        # impossible to inspect while it is going.
        if chapter_flags:
            with (work / "flagged.txt").open("a", encoding="utf-8") as fh:
                fh.write("\n".join(chapter_flags) + "\n")

        print(f"  ch{ch.index:3} {len(chunks):4} chunks  {mastered.seconds/60:5.1f} min  "
              f"{time.perf_counter()-t0:6.1f}s  rss {rss_gb:4.1f}G  "
              f"RMS {mastered.stats['rms_after_dbfs']:5.1f} "
              f"peak {mastered.stats['peak_after_dbfs']:5.1f} dBFS"
              + (f"  [{retried} retries]" if retried else "")
              + (f"  [{bad} flagged]" if bad else ""))

        # Release the chapter's audio before starting the next one, AFTER the
        # line above has read it. Measured on a 10-chapter book in one process:
        # RSS climbed 15.3 -> 22.2 GB with current within 2% of peak, so nothing
        # was being returned. This bounds the chapter-local buffers; the leak
        # itself is per chunk, so a long run also needs the driver's restart.
        pieces.clear()
        mastered = None
        gc.collect()
        if backend in ("espeech", "chatterbox"):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

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
    i.add_argument("--latin", default="route", choices=normalise.LATIN_MODES)
    i.add_argument("-v", "--verbose", action="store_true")
    i.set_defaults(fn=cmd_inspect)

    e = sub.add_parser("edit", help="Claude's read-through of a Russian book: "
                                     "stress, ё and pauses, cached for convert --editor claude")
    e.add_argument("epub")
    e.add_argument("--chapters", default=None, help='"1", "1-3", "1,4,7-9"')
    e.add_argument("--lang", default=None)
    e.add_argument("--work", default="work")
    e.add_argument("--max-chars", type=int, default=normalise.MAX_CHUNK_CHARS)
    e.add_argument("--latin", default="route", choices=normalise.LATIN_MODES)
    e.add_argument("--editor-model", default=llm_edit.DEFAULT_MODEL)
    e.add_argument("--editor-jobs", type=int, default=llm_edit.JOBS)
    e.add_argument("--force", action="store_true", help="ignore cached edits")
    e.set_defaults(fn=cmd_edit)

    c = sub.add_parser("convert", help="convert an EPUB to an M4B")
    c.add_argument("epub")
    c.add_argument("--backend", default=None,
                   choices=("silero", "kokoro", "espeech", "chatterbox",
                            "higgs", "omnivoice", "piper", "qwen-stress", "pocket"),
                   help="default: chosen per language (ru->silero, en->kokoro)")
    c.add_argument("--stress", default="ruaccent",
                   choices=("ruaccent", "silero", "none"),
                   help="Russian stress marker; silero also restores ё")
    c.add_argument("--variant", default="rlv2",
                   choices=("rlv1", "rlv2", "sft95k", "podcaster", "sft", "slow", "fast"),
                   help="espeech: which checkpoint; pocket: slow (default) or fast delivery")
    c.add_argument("--latin", default="route", choices=normalise.LATIN_MODES,
                   help="Latin script in a Russian book: route English sentences to an "
                        "English voice and transliterate Latin words in Russian ones "
                        "(default); translit everything into Cyrillic; or keep it as is")
    c.add_argument("--en-backend", default="pocket", choices=("pocket", "qwen-stress"),
                   help="pocket: which model reads the English sentences -- Kyutai's "
                        "English Pocket (default) or the qwen-stress clone of --voice")
    c.add_argument("--en-voice", default=None,
                   help="pocket: voice for the English sentences -- a clip (default: "
                        "--voice, Kyutai's gated cloning model) or a Kyutai preset name "
                        "such as george, michael, peter_yearsley")
    c.add_argument("--device", default="auto",
                   help="cuda | mps | cpu | auto (default: detect)")
    c.add_argument("--speed", type=float, default=None,
                   help="espeech only: 1.0 is ~21 chars/s, 0.82 is ~18")
    c.add_argument("--voice", default=None,
                   help="silero speaker / kokoro voice, or reference wav for "
                        "chatterbox, espeech, qwen-stress and pocket")
    c.add_argument("--lang", default=None, help="override the EPUB's dc:language")
    c.add_argument("--out", default=None)
    c.add_argument("--work", default="work")
    c.add_argument("--limit", type=int, default=0, help="first N chapters only")
    c.add_argument("--chapters", default=None,
                   help="render only these chapters: \"21\", \"21-25\", \"1,3,7-9\"")
    c.add_argument("--editor", default="none", choices=("none", "claude"),
                   help="have Claude read the chapter first and fix stress, ё and "
                        "pauses (pocket / qwen-stress only; cached under work/<book>/edits)")
    c.add_argument("--editor-model", default=llm_edit.DEFAULT_MODEL,
                   help=f"model for --editor claude (default {llm_edit.DEFAULT_MODEL})")
    c.add_argument("--editor-jobs", type=int, default=llm_edit.JOBS,
                   help="parallel claude -p calls")
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


if __name__ == "__main__":  # `python -m ebooker.cli`
    import sys
    sys.exit(main(sys.argv[1:]))

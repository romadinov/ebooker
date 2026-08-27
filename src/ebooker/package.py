"""Mastered chapters -> a single .m4b that Apple Books treats as an audiobook.

An M4B is an MP4 with chapter atoms plus `stik=2`, the flag that makes Books
file it as an audiobook (and so remember playback position) instead of as music.

Verified on this machine: ffmpeg writes `stik` on its own via
`-metadata media_type=2`, so AtomicParsley / mp4v2 / m4b-tool are not required.

Delivery caveat worth knowing before building a library around Apple Books:
iCloud syncs *ebooks* only. A sideloaded audiobook imports into Books on macOS
with chapters intact, but does not sync to iPhone/iPad and does not sync
playback position. Getting it onto a phone means a Finder cable sync. The output
here is standard M4B, so Audiobookshelf remains a drop-in alternative.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# 64 kbps mono AAC is about what commercial audiobooks ship at, and is
# transparent for speech. aac_at is Apple's AudioToolbox encoder (better than
# ffmpeg's native aac); fall back if unavailable.
BITRATE = "64k"
SAMPLE_RATE = 44100


@dataclass
class ChapterSpec:
    title: str
    path: Path          # mastered chapter wav
    seconds: float


def _have_encoder(name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                         capture_output=True, text=True).stdout
    return any(line.split()[1] == name
               for line in out.splitlines() if len(line.split()) > 1)


def _ffescape(s: str) -> str:
    """ffmetadata escaping: =, ;, #, \\ and newlines are special."""
    for ch in ("\\", "=", ";", "#"):
        s = s.replace(ch, "\\" + ch)
    return s.replace("\n", " ")


def write_ffmetadata(chapters: list[ChapterSpec], meta: dict, dest: Path) -> Path:
    lines = [";FFMETADATA1"]
    for key in ("title", "artist", "album", "album_artist", "composer",
                "genre", "date", "comment", "description"):
        if meta.get(key):
            lines.append(f"{key}={_ffescape(str(meta[key]))}")
    t = 0.0
    for ch in chapters:
        start = int(round(t * 1000))
        t += ch.seconds
        end = int(round(t * 1000))
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}",
                  f"END={end}", f"title={_ffescape(ch.title)}"]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def build(
    chapters: list[ChapterSpec], meta: dict, out_path: Path,
    cover: Path | None = None, work_dir: Path | None = None,
    bitrate: str = BITRATE, sample_rate: int = SAMPLE_RATE,
) -> Path:
    if not chapters:
        raise ValueError("no chapters to package")
    work = work_dir or out_path.parent
    work.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    concat = work / "_concat.txt"
    concat.write_text(
        "".join(f"file '{c.path.resolve().as_posix()}'\n" for c in chapters),
        encoding="utf-8")
    ffmeta = write_ffmetadata(chapters, meta, work / "_ffmeta.txt")

    codec = "aac_at" if _have_encoder("aac_at") else "aac"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "concat", "-safe", "0", "-i", str(concat),
           "-i", str(ffmeta)]
    if cover and cover.exists():
        cmd += ["-i", str(cover)]

    cmd += ["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]
    if cover and cover.exists():
        # Cover rides as a still image stream flagged as attached art.
        cmd += ["-map", "2:v", "-c:v", "mjpeg", "-disposition:v:0", "attached_pic"]
    cmd += ["-c:a", codec, "-b:a", bitrate, "-ac", "1", "-ar", str(sample_rate),
            "-metadata", "media_type=2",          # stik=2 -> audiobook
            "-metadata", "genre=Audiobook",
            "-brand", "M4B ",
            "-movflags", "+faststart",
            "-f", "ipod", str(out_path)]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{r.stderr[-2000:]}")
    concat.unlink(missing_ok=True)
    return out_path


def probe(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-hide_banner", "-loglevel", "error", "-of", "json",
         "-show_format", "-show_streams", "-show_chapters", str(path)],
        capture_output=True, text=True)
    return json.loads(r.stdout or "{}")


def verify_m4b(path: Path) -> tuple[bool, list[str]]:
    """Confirm the file is actually an audiobook, not just named like one."""
    info = probe(path)
    problems: list[str] = []
    fmt = info.get("format", {})
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}

    if str(tags.get("media_type", "")) != "2":
        problems.append("stik/media_type != 2 -- Books will treat this as music")
    if not info.get("chapters"):
        problems.append("no chapters")
    audio = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        problems.append("no audio stream")
    elif audio[0].get("codec_name") != "aac":
        problems.append(f"audio codec is {audio[0].get('codec_name')}, expected aac")
    if not tags.get("title"):
        problems.append("no title tag")
    if not any(s.get("codec_type") == "video" for s in info.get("streams", [])):
        problems.append("no cover art")
    return not problems, problems

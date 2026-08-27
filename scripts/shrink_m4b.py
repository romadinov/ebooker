"""Re-encode an M4B to fit a size limit, keeping chapters, cover and metadata.

Sharing services cap uploads (25 MB is common), and a 64 kbps hour of speech is
about 28 MB, so a two-chapter sample already overflows. Bitrate is the only
lever worth pulling: chapter markers, cover art and tags are tiny, and dropping
them would not help.

Apple's AudioToolbox encoder in this ffmpeg build has no HE-AAC profile
(`-profile:a aac_he` is rejected), so this is plain AAC-LC. 48 kbps mono is
transparent enough for narration -- it is what the showcase page embeds.

    shrink_m4b.py <in.m4b> [out.m4b] [--limit-mb 25] [--min-bitrate 32]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


def probe(path: pathlib.Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-of", "json", "-show_format",
                        "-show_streams", "-show_chapters", str(path)],
                       capture_output=True, text=True)
    return json.loads(r.stdout or "{}")


def have_encoder(name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                         capture_output=True, text=True).stdout
    return any(len(l.split()) > 1 and l.split()[1] == name for l in out.splitlines())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dest", nargs="?")
    ap.add_argument("--limit-mb", type=float, default=25.0)
    ap.add_argument("--min-bitrate", type=int, default=32,
                    help="kbps floor; below this speech starts to sound thin")
    a = ap.parse_args()

    src = pathlib.Path(a.src)
    if not src.exists():
        print(f"no such file: {src}", file=sys.stderr)
        return 1
    info = probe(src)
    dur = float(info.get("format", {}).get("duration", 0) or 0)
    if dur <= 0:
        print("could not read duration", file=sys.stderr)
        return 1
    size_mb = src.stat().st_size / 1048576
    chapters = info.get("chapters", [])
    has_cover = any(s.get("codec_type") == "video" for s in info.get("streams", []))
    tags = info.get("format", {}).get("tags", {}) or {}

    print(f"{src.name}: {size_mb:.1f} MB, {dur/60:.1f} min, "
          f"{len(chapters)} chapter(s), cover {'yes' if has_cover else 'no'}")
    if size_mb <= a.limit_mb:
        print(f"already under {a.limit_mb:.0f} MB; nothing to do")
        return 0

    # Reserve headroom for container overhead, chapter track and cover art.
    overhead_mb = 0.4 + (0.25 if has_cover else 0.0)
    budget_bits = (a.limit_mb - overhead_mb) * 1048576 * 8
    wanted = int(budget_bits / dur / 1000 * 0.97)    # 3% safety margin

    # AudioToolbox quantises to a fixed AAC-LC ladder and rounds *up* to the
    # nearest allowed rate, so asking for 58 kbps silently produced 64 and the
    # file did not shrink at all. Snap down to a rate the encoder will accept.
    LADDER = [32, 40, 48, 56, 64]
    allowed = [b for b in LADDER if b <= wanted and b >= a.min_bitrate]
    kbps = max(allowed) if allowed else a.min_bitrate
    print(f"target {a.limit_mb:.0f} MB -> wanted {wanted} kbps, "
          f"using {kbps} kbps mono (encoder ladder)")

    dest = pathlib.Path(a.dest) if a.dest else src.with_name(f"{src.stem}_small.m4b")
    codec = "aac_at" if have_encoder("aac_at") else "aac"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
           "-map", "0:a:0", "-map_metadata", "0", "-map_chapters", "0"]
    if has_cover:
        # Re-encode the art rather than copying: some builds refuse to remux an
        # attached picture into the ipod container.
        cmd += ["-map", "0:v:0", "-c:v", "mjpeg", "-disposition:v:0", "attached_pic"]
    cmd += ["-c:a", codec, "-b:a", f"{kbps}k", "-ac", "1", "-ar", "44100",
            "-metadata", "media_type=2", "-brand", "M4B ",
            "-movflags", "+faststart", "-f", "ipod", str(dest)]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ffmpeg failed:\n{r.stderr[-1500:]}", file=sys.stderr)
        return 1

    out_info = probe(dest)
    out_mb = dest.stat().st_size / 1048576
    out_ch = len(out_info.get("chapters", []))
    out_tags = {k.lower(): v for k, v in
                (out_info.get("format", {}).get("tags", {}) or {}).items()}
    out_cover = any(s.get("codec_type") == "video"
                    for s in out_info.get("streams", []))
    print(f"\n{dest}  {out_mb:.1f} MB  ({size_mb/out_mb:.1f}x smaller)")
    print(f"  chapters {out_ch}/{len(chapters)}   "
          f"cover {'kept' if out_cover == has_cover else 'LOST'}   "
          f"audiobook flag {'set' if str(out_tags.get('media_type','')) == '2' else 'MISSING'}")
    print(f"  title  {out_tags.get('title', tags.get('title', '(none)'))}")
    if out_mb > a.limit_mb:
        print(f"  ! still over {a.limit_mb:.0f} MB", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ebooker

Turn a shelf of EPUBs into Apple-compatible M4B audiobooks, locally.

Runs on an Apple Silicon Mac or an NVIDIA box (tested on a DGX Spark / GB10) with
the same command — the device is detected, not configured. Every chunk of
generated speech is verified by transcribing it back and comparing to the source,
because these models drop and mangle words silently.

```bash
uv run ebooker inspect  book.epub      # what's in it, how long it will be
uv run ebooker convert  book.epub      # -> out/<Title>.m4b
```

## Why the verification matters

Neural TTS fails quietly. It drops a numeral, merges two words, reads a
quotation mark as a two-second pause, or mispronounces a name — and none of that
raises an error. On a 10-hour book that is ~5,000 chunks, far too many to
audition. So every chunk is transcribed with Whisper and compared to what was
sent, and anything suspect is re-generated with a different seed. What survives
three attempts goes to a review list rather than silently into the book.

The checks are deliberately not just a character error rate, because averages
hide short errors in long chunks:

- **numeral integrity** — a missing number is a factual change to the book
  ("sixteen passengers died" became "sew passengers died" in testing)
- **word integrity** — content words must survive, matched on stems so Russian
  case endings and British/American spelling don't raise false alarms
- **word-count deficit** — catches words the model merged into a neighbour
- **duration, silence, clipping, tail-repetition** — cheap signal checks needing
  no model

## Install

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and ffmpeg.

```bash
git clone https://github.com/romadinov/ebooker && cd ebooker
uv sync
```

Backends live in separate environments where their pins conflict — see
[docs/BACKENDS.md](docs/BACKENDS.md).

## Backends

Chosen per language automatically. Every figure below is measured, not
estimated; RTF is the real-time factor, so 0.5 means twice as fast as playback.

| backend | language | error rate | RTF (M4 Max) | RTF (GB10) | notes |
|---|---|---|---|---|---|
| **Silero v5** | ru, uk, be | 1.7% | **0.008** | — | native stress + homographs, fixed voices |
| **Kokoro-82M** | en + 7 more | **0.0%** | 0.055 | — | Apache 2.0, no cloning |
| **ESpeech-TTS-1** | ru | 0.7–1.6% | 0.78 | **0.25** | F5/DiT, voice cloning, honours stress marks |
| Chatterbox | 23 langs | 3.3% (ru) | 2.7–3.9 | — | MIT, cloning; **ignores stress marks** |

Override with `--backend`, pick a voice with `--voice`, and see
`ebooker convert --help` for the rest.

## Russian stress

Russian stress carries meaning — *за́мок* is a castle, *замо́к* is a lock — and
a speech model that guesses wrong is simply reading a different word. ebooker
marks stress explicitly with [RUAccent](https://github.com/Den4ikAI/ruaccent)
before synthesis rather than trusting the model.

Two editable files carry corrections, because you will find more by ear:

- **`stress_overrides.txt`** — `word = m+arked`, for words the accentuator gets
  wrong (its dictionary maps *после* to the incorrect *посл+е*)
- **`pronunciation.txt`** — `written = respelled`, for words read wrong at the
  grapheme level. Transliterated foreign names are the usual offenders: *Иоган*
  came out as "и ганг", while *Йоган* is said correctly. Stress marks cannot fix
  this; only respelling can.

Not every model accepts stress marks. Silero and ESpeech take `+` before the
stressed vowel. Chatterbox is destroyed by it (58.5% error rate — it reads the
plus signs aloud) and tolerates the combining acute U+0301 while **not actually
honouring it**, which is why it is not the Russian default.

## Mac and NVIDIA

`--device auto` (the default) picks CUDA, then MPS, then CPU. The same command
works on both.

One caveat the tool will warn you about: on Apple Silicon, Metal's MPSGraph
compilation cache is keyed on tensor shape, has no eviction path, and lives for
the life of the process ([pytorch#181213](https://github.com/pytorch/pytorch/issues/181213)).
Flow-matching TTS produces a different shape for nearly every chunk, so the
cache grows until the process stops making progress — measured here as a slide
from 10 to 31 minutes per chapter and then a stall. Only a restart clears it, so
on MPS render a chapter per process:

```bash
for n in $(seq 1 53); do
  uv run ebooker convert book.epub --chapters $n --render-only
done
uv run ebooker convert book.epub --package-only
```

CUDA has no such problem — run it in one go. Rendering and packaging are
separable, so a book can be rendered on one machine and assembled on another.

## Output

A single M4B with chapter markers from the book's own table of contents, cover
art, and `stik=2` — the atom that makes Apple Books treat the file as an
audiobook and remember your position. Levels are mastered to the ACX window
(RMS −23 to −18 dBFS, true peak ≤ −3, noise floor ≤ −60) which Apple accepts.

**Apple Books caveat:** iCloud syncs *ebooks* only. A sideloaded audiobook
imports on macOS with chapters intact but does not sync to iPhone or iPad, and
playback position does not follow you across devices. Getting it onto a phone
means a Finder cable sync. The output is standard M4B, so
[Audiobookshelf](https://www.audiobookshelf.org/) works as an alternative front
end with no regeneration.

## Licence

This code is MIT — see [LICENSE](LICENSE).

The models are not, and their terms are yours to observe:

| model | licence |
|---|---|
| Kokoro-82M | Apache 2.0 |
| ESpeech-TTS-1 | Apache 2.0 |
| Chatterbox | MIT (embeds a [PerTh](https://github.com/resemble-ai/perth) watermark) |
| Silero | main models **CC-BY-NC**; base CIS models MIT |
| RUAccent | see upstream |
| Whisper | MIT |

**No books, audio or voice references are included in this repository**, and
none should be added to it. Cloning a real narrator's voice from a commercial
audiobook is a thing this tool can technically do; publishing the result is
not something it should be used for.

## Further reading

[DESIGN.md](DESIGN.md) is the build log — what was measured, what turned out to
be wrong, and why the defaults are what they are.

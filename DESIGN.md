# ebooker — EPUB → Apple audiobook (M4B) pipeline

Design doc. Target: narrator-grade M4B audiobooks from a mixed Russian/English
EPUB library, generated locally, at effectively zero marginal cost.

---

## 1. Verdict up front

**Model: Chatterbox Multilingual V3** (Resemble AI, 0.5B, MIT licence) as the single
TTS backend for both languages. It covers Russian *and* English among its 23
languages, does zero-shot voice cloning from ~5 s of reference audio, and won
65.3% of blind A/B tests against ElevenLabs. One cloned voice, both languages,
one code path — this is the decision that collapses most of the complexity.

**Machines: Spark generates, MBP orchestrates and verifies.** Chatterbox's MPS
support on Apple Silicon is unreliable and commonly falls back to CPU, so the
M4 Max is a poor synthesis box for this model. The Spark's GB10 runs it properly.
Meanwhile the M4 Max is excellent at `whisper.cpp`, which is exactly what the QC
stage needs — so the two machines pipeline against each other instead of idling.

**Cost: local wins by roughly three orders of magnitude.** Not a close call:

| Path | This book (745k chars) | 100-book library (~50M chars) |
|---|---|---|
| ElevenLabs Multilingual v3 ($100/1M) | ~$75 | ~$5,000 |
| ElevenLabs Flash/Turbo ($50/1M) | ~$37 | ~$2,500 |
| Local (Spark @ ~240 W) | a few cents of power | ~$20–40 of power |

The only real cost is your setup time — budget 1–2 days, most of it fighting
the Spark's aarch64/CUDA-13 wheel situation (§6). After that, marginal cost per
book is electricity.

---

## 2. What the sample book tells us

`Garrison_Plenennaya-Vselennaya.517107.fb2.epub` — a 3-novel Harry Harrison
collection, `dc:language = ru`, converted from FB2:

- **745k characters**, ~124k words → **~13.8 h of audio** at 150 wpm
- 48 XHTML files, **60 `navPoint`s** in `toc.ncx` — but the TOC is mostly bare
  numbers (`1`, `2`, `3`…) nested under real titles (`Врач космического корабля`)
- FB2→EPUB conversion artifacts: embedded Times New Roman fonts, 10 inline images

Three consequences that shape the design:

1. **Books are long.** ~14 h of audio ≈ 2,500–3,700 synthesis chunks. Any pipeline
   without resumable per-chunk state will lose hours of work to one crash. State
   tracking is not a nice-to-have.
2. **Chapter structure needs flattening logic.** Nested numeric navPoints must
   become sensible chapter names (`Врач космического корабля — 1`), or the M4B
   chapter list is 60 entries called "1" through "45".
3. **FB2-origin EPUBs are dirty.** Expect inconsistent markup across the library;
   the ingest stage must be defensive, not clever.

---

## 3. Pipeline architecture

Six stages, each writing to disk, each independently resumable. A per-book
SQLite manifest is the single source of truth for chunk state.

```
  EPUB
    │
    ▼
┌───────────────────┐
│ 1. INGEST         │  spine + NCX → ordered chapters, titles, cover, dc:language
│    (MBP, cheap)   │  strip fonts/images/footnote markers/front matter
└─────────┬─────────┘  → book.json
          ▼
┌───────────────────┐
│ 2. NORMALISE      │  numbers/dates/abbrevs → words (num2words, case-aware for RU)
│    (MBP, cheap)   │  RU: ё-restoration, homograph stress via RUAccent
│                   │  sentence-segment → chunks ≤250 chars, never mid-sentence
└─────────┬─────────┘  → chunks in manifest.db (status=pending)
          ▼
┌───────────────────┐
│ 3. SYNTHESISE     │  Chatterbox Multilingual V3, N concurrent workers
│    (SPARK, GPU)   │  fixed voice embedding per book, seed recorded per chunk
└─────────┬─────────┘  → chunk_0001.wav … (status=synthesised)
          ▼
┌───────────────────┐
│ 4. VERIFY         │  whisper.cpp ASR each chunk → normalise → CER vs source text
│    (MBP, ANE/GPU) │  fail ⇒ requeue to stage 3 with new seed / smaller chunk
└─────────┬─────────┘  → status=verified | needs_review
          ▼
┌───────────────────┐
│ 5. MASTER         │  trim, pause insertion, ACX-compliant loudness, concat/chapter
│    (either)       │  → chapter_01.wav …
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 6. PACKAGE        │  AAC-LC 64k mono → M4B + chapters + cover + stik=2
│    (MBP)          │  → Book Title.m4b
└───────────────────┘
```

Stages 3 and 4 run **concurrently against each other**: the Spark synthesises
chunk N+1 while the MBP verifies chunk N. Verification is not a serial tax.

---

## 4. The quality engineering

This section is the difference between "a robot read my book" and "narrator-grade".
The model choice matters less than these five things.

### 4.1 Text normalisation is the cheapest quality win available

Every TTS model mangles unexpanded text. Before synthesis:

- **Numbers, dates, ordinals, currency, roman numerals** → spelled-out words.
  For Russian this must be **case- and gender-aware** (`в 1991 году` →
  `в тысяча девятьсот девяносто первом году`, not the nominative form). `num2words`
  gets you started but needs a grammatical-case wrapper.
- **Abbreviations** → expansions (`т.е.` → `то есть`, `др.` → `другие`).
- **Dialogue em-dashes** — Russian convention (`— Привет,`) must not be read as
  a pause artifact or a minus sign.
- **Footnote markers, page numbers, image captions** → deleted, not read aloud.
- **Latin fragments inside Russian text** — either transliterate or accept the
  model's accent; decide once, apply consistently.

### 4.2 Russian stress and homographs

Russian stress is meaning-bearing (`за́мок` castle vs `замо́к` lock). Chatterbox
has no explicit stress control. Two mitigations, in order:

1. **Test whether RUAccent stress marks help.** Inject `+` before stressed vowels
   and A/B a chapter. The model may honour them, ignore them, or read them aloud —
   this is empirical, not predictable. Test before committing.
2. **If marks don't work**, fall back to a per-book pronunciation dictionary of
   targeted respellings for the handful of homographs that actually recur.

**Fallback worth knowing about:** Silero v5 (`v5_5_ru`) has *native* automated
stress and homograph resolution, SSML, and an RTF of ~0.04 on CPU — 24× faster
than real time, no GPU at all. Its prosody is flatter than Chatterbox's, so it's
not first choice for narrator-grade output, but if Chatterbox's Russian stress
errors prove unfixable, Silero is the escape hatch. Bake-off both in Phase 0.

### 4.3 Chunking

- **Never split mid-sentence.** Chunk on sentence boundaries, packing to ≤250
  characters (~17 s of speech). Chatterbox degrades toward the end of long
  generations, and flow-matching models like F5 hard-cap around 30 s.
- Carry paragraph and chapter boundaries as metadata so stage 5 can insert the
  right pauses.

### 4.4 ASR round-trip verification — the core reliability mechanism

Autoregressive TTS **silently drops and hallucinates words**; F5-TTS is documented
to skip words on repeated tokens, and Chatterbox is not immune. At 3,000 chunks
per book you cannot listen to it all. So verify mechanically:

1. Transcribe each generated chunk with `whisper.cpp` (large-v3-turbo).
2. Normalise both the ASR output and the source chunk (casefold, strip punctuation,
   spell out digits identically).
3. Compute **CER**. Flag above threshold (start ~8%, tune per language — Russian
   ASR will be noisier than English, so use separate thresholds).
4. **Retry with a different seed.** Up to 3 attempts, then bisect the chunk into
   smaller pieces, then mark `needs_review`.

Also flag mechanically, no ASR needed:
- **Duration outliers** — chars/second far off the book's median means a stall or a loop.
- **Silence-only or near-silent output.**
- **End-of-chunk repetition loops** — detect via autocorrelation on the tail.
- **Clipping.**

Note that Whisper itself hallucinates on inputs beyond its 30 s receptive field —
another reason to keep chunks under ~20 s, which the chunker already does.

### 4.5 Pacing and mastering

Machine-gun delivery is the most common tell of an AI audiobook. Insert real silence:

- inter-sentence: ~350 ms
- inter-paragraph: ~700 ms
- scene break / chapter head and tail: ~1 s

Then master to the **ACX spec**, which Apple Books accepts and which is the
industry norm for spoken word — all three must hold simultaneously:

| Measure | Target |
|---|---|
| RMS | between **−23 and −18 dBFS** (aim −20) |
| True peak | **≤ −3 dBFS** |
| Noise floor | **≤ −60 dBFS** |

Watch the interaction: gain applied to reach the RMS window lifts the noise floor
by the same amount. If you need 8 dB of gain, the raw noise floor must start below
−68 dBFS. Neural TTS output is usually clean enough that this is fine, but measure
rather than assume.

---

## 5. Apple delivery — and one significant gotcha

**Packaging.** M4B is just MP4 with chapter atoms and a bookmarking flag:

- **Codec:** AAC-LC, **64 kbps mono @ 44.1 kHz** is ample for speech (roughly what
  Audible ships). A 14 h book lands around 400 MB.
- **Chapters:** ffmpeg `ffmetadata` chapter markers derived from the flattened TOC.
- **`stik` atom = 2** — this is what makes Books treat the file as an *audiobook*
  rather than music, and what enables position memory. Set via AtomicParsley.
- **Metadata:** title, author, narrator, year, genre, plus the cover art extracted
  during ingest.
- Easiest path: **`m4b-tool`** wraps ffmpeg + mp4v2 and handles the atom details.
  Roll your own with ffmpeg + AtomicParsley only if you want the control.

### ⚠️ Apple Books does not iCloud-sync sideloaded audiobooks

This is worth knowing *before* you build around Apple Books. Verified behaviour:

- Books on **Mac** imports M4B correctly, chapters intact. ✅
- **iCloud sync covers ebooks only** — imported audiobooks do not sync to iPhone/iPad.
- Getting them onto a phone requires **Finder cable sync**.
- **Playback position does not sync across devices** for sideloaded audiobooks.

If you listen on one device, Apple Books is fine. If you expected phone/Mac
handoff, it won't happen, and that's an Apple restriction, not something the
pipeline can fix. The pipeline should emit standard M4B regardless — that keeps
**Audiobookshelf** (self-hosted, syncs position across everything) available as a
drop-in alternative front end without regenerating anything.

---

## 6. Hardware split, and the Spark setup tax

| Role | Machine | Why |
|---|---|---|
| Ingest, normalise, package | MBP (M4 Max, 64 GB) | Cheap, interactive, tooling already installed |
| **Synthesis** | **DGX Spark (GB10, 128 GB)** | Chatterbox MPS support on Apple Silicon is unreliable and falls back to CPU |
| **ASR verification** | **MBP** | `whisper.cpp` is excellent on Apple Silicon; keeps the GPU free for synthesis |

**The Spark's real constraint is bandwidth, not compute.** GB10 offers ~1 PFLOP
of sparse FP4, but its 128 GB of unified LPDDR5x runs at **273 GB/s** — that's the
bottleneck for inference, and it's well below a discrete workstation GPU. For a
0.5B model like Chatterbox this is survivable, and the win comes from running
**several concurrent synthesis workers** rather than from single-stream speed.

**Expect friction getting there.** The aarch64 + CUDA 13 + sm_121 combination is
still rough:

- Default PyPI PyTorch wheels are built against CUDA 12 → `libcudart.so.12: cannot
  open shared object file`. Install from the **cu130 index**.
- Many prebuilt wheels compile kernels only through sm_120. Useful fact:
  **sm_120 and sm_121 are binary compatible** (confirmed by PyTorch maintainers),
  so an sm_120 build generally runs on GB10 — the failures are usually CUDA-major
  mismatches or missing aarch64 builds, not the architecture itself.
- Community prebuilt aarch64/sm_121 wheel repos exist and will save you hours.
- Prefer **NGC containers** where possible over building the environment by hand.

**Do not skip Phase 0 benchmarking.** I have no measured Chatterbox-on-GB10
throughput figure, and I'm not going to invent one. Measure single-stream RTF,
then find the concurrency level where aggregate throughput plateaus (bandwidth
saturation will cap it well before core count does). That number sets your
per-book wall-clock, and everything downstream is planning fiction until you have it.

---

## 6a. Phase 0 results — measured, 2026-08-26

Run on the M4 Max (64 GB). 8 test cases drawn from the real book (narration,
dialogue, a 17-char utterance, decimals, an alphanumeric designator, a homograph
torture test, a transliterated-Nahuatl epigraph) x 6 backends = 48 clips, scored
by transcribing each with whisper-large-v3-turbo and comparing to the input.

| backend | median CER | max CER | median RTF | est. full book (13.6 h) |
|---|---|---|---|---|
| **silero-aidar** | **0.0%** | 37.0% | **0.0066** | **5 min** |
| **silero-baya** | **0.0%** | 20.4% | **0.0066** | **5 min** |
| **silero-eugene** | **0.0%** | 35.2% | 0.0073 | **6 min** |
| silero-xenia | 0.6% | 31.5% | 0.0072 | 6 min |
| chatterbox-cloned | 3.3% | 16.7% | 3.95 | 53.7 h |
| chatterbox-default | 4.3% | 27.8% | 2.48 | 33.7 h |

Every remaining flag on the leading backends is the Nahuatl epigraph, which is
unverifiable by ASR rather than badly spoken (see below).

**The Phase 0 bet did not pay off the way the design assumed.** Chatterbox was
picked as the primary on the strength of its English blind-test record and its
single-model coverage of both languages. On *Russian* it is measurably worse
than Silero on intelligibility and between 340x and 540x slower. Specifically:

* **Silero resolves stress and ё natively, and it works.** On the homograph
  test it scored 0.0% and returned "Лёва Королёв" with ё restored and
  готов/готов correctly disambiguated. This is the thing the design was most
  worried about, and it is simply handled -- no RUAccent, no pronunciation
  dictionary. The RUAccent experiment is therefore moot for Silero.
* **Chatterbox mishandles very short utterances.** On "— Что за тревога?"
  (17 chars) it produced 5.1 s of audio for ~1.1 s of speech and said
  "певага" -- a duration outlier plus a real word error. 12.5% of this book's
  chunks are under 40 chars, mostly dialogue turns, so this is not an edge case.
  Supplying a reference voice largely fixes it (1.4 s, 7.1% CER), so cloning is
  not optional for Chatterbox on dialogue-heavy fiction.
* **Silero voices are not interchangeable.** `xenia` failed the homograph test
  at 25.3% and appended hallucinated text, while `aidar`/`baya`/`eugene` scored
  0.0%. Voice choice needs its own check, not just a taste test.

**Revised recommendation: Silero is the default for Russian**, with Chatterbox
reserved for (a) English, where its blind-test record is the relevant evidence
and Silero has no voice, and (b) any book where one cloned voice across both
languages matters more than 340x throughput. That makes the DGX Spark optional
rather than load-bearing for the Russian majority of the library: the whole book
renders in ~6 minutes of synthesis on the laptop, ~24 minutes more for full ASR
verification. Benchmark Chatterbox on the Spark before committing to it for the
English shelf.

### Corrections to earlier assumptions

* **ffmpeg alone writes `stik`.** `-metadata media_type=2` with `-f ipod`
  produces a real audiobook atom -- verified by hexdump and ffprobe.
  AtomicParsley / mp4v2 / m4b-tool are **not** required. Setting
  `-brand "M4B "` works too.
* **MPS is slower, not broken.** Chatterbox on MPS ran at RTF 5.62 versus 3.08
  on CPU for the same input. Earlier reports of MPS crashes no longer reproduce;
  there is simply nothing to gain by enabling it on Apple Silicon.
* **`perth` needs `setuptools<81`.** The PerTh watermarker imports
  `pkg_resources`; without it, perth's `__init__` swallows the ImportError and
  leaves the class as `None`, surfacing as
  `TypeError: 'NoneType' object is not callable` at model load. The Spark will
  hit this too.
* **Intermediate WAVs must be float.** ~5% of Silero chunks exceed full scale
  (peaks to 1.25). `soundfile` defaults `.wav` to PCM_16, which clamps them on
  write and makes the clipping unrepairable. Write `subtype="FLOAT"`, and apply
  per-chunk headroom *before* concatenation.
* **The ASR verifier needs number canonicalisation.** Whisper writes numerals as
  digits, so a spelled-out source ("две целых восемь десятых") scores 33% CER
  against a perfect rendering transcribed as "2,8". Decimals must be expanded
  before integers. Without this fix every numeric chunk in the library is a
  false positive.
* **Some text is unverifiable by ASR.** The book's Nahuatl and Otomi epigraphs
  transcribe badly however cleanly they are spoken. These are separated from
  real defects by retrying with different seeds: stochastic failures move, and a
  CER that stays high *and* nearly identical across seeds is evidence about the
  text, not the audio. Those go to human review instead of consuming retries.

## 6b. Phase 0 results — English, measured 2026-08-26

Test book: *Old Man's War*, John Scalzi (`dc:language=en`), 505k chars, 9.4 h,
19 chapters. 6 cases x 6 backends = 36 clips, same scoring method.

| backend | median CER | median RTF | est. 9.4 h book | clipped |
|---|---|---|---|---|
| **kokoro-am_michael** | 0.0% | **0.083** | **47 min** | 0 |
| kokoro-bm_george | 0.0% | 0.085 | 48 min | 0 |
| kokoro-af_heart | 0.0% | 0.111 | 63 min | 0 |
| chatterbox-en-turbo | 0.0% | 0.99 | 9.3 h | 0 |
| chatterbox-en-base | 0.0% | 2.77 | 26.0 h | 2 |
| chatterbox-mtl-en | 0.0% | 3.99 | 37.5 h | 1 |

**Every backend scored 0.0% median CER with zero flags.** English is simply
easier than Russian: no stress to place, no homographs to disambiguate, and
ASR agrees with all six models. CER cannot discriminate here, so **cost
decides, and Kokoro wins it by 12x to 48x.**

Three secondary findings:

* **Chatterbox Multilingual is the worst choice for English** -- slowest of the
  six and no more intelligible. The single-model-for-both-languages idea costs
  something on *each* language rather than paying off on either.
* **Chatterbox-Turbo is the interesting middle**: 2.8x faster than the English
  base model at identical CER, and the only Chatterbox variant whose CPU
  RTF (~1.0) is even close to practical. If the Spark is going to earn its
  place, Turbo is the workload to benchmark there.
* **The short-utterance defect was Russian-specific.** Chatterbox mangled a
  17-char Russian line (5.1 s of audio, wrong word); on English the same class
  of input was fine across all variants.

### English needs *more* text normalisation than Russian, not less

This was the opposite of the expectation. Measured on the Scalzi:

* **207 acronym expansions.** "CDF" alone occurs **151 times**, plus PDA (53),
  MP (20), DNA (16), CDFS. Unexpanded, these are gibberish.
* **10 military times** (`0600` must be "oh six hundred", not "six hundred").
* **Serial numbers** (`ASD324DDD4E3C1`) must read digit-by-digit; "three
  hundred and twenty-four" is wrong for a serial.
* 79 over-long sentences hard-split, versus 15 in the Russian book.

Compare the Russian book: 156 numeric tokens in 738k chars and **zero** Latin
runs. The Russian difficulty is phonetic (stress, homographs) and the model
handles it; the English difficulty is orthographic and the *pipeline* must
handle it.

Acronym detection asks "is this a real English word?" rather than "is this
uppercase?", because `ONE`/`TWO` chapter titles and `NEW`/`YOU`/`BODY`/`YES`
emphasis all appear fully capitalised — spelling those out would be worse than
leaving acronyms alone. Roman numerals are matched only from a closed safe set:
the general solution is actively harmful, since `CD`, `MD`, `DC`, `MC`, `MI` and
`XL` are all valid roman numerals *and* plausible acronyms, and `ID` parses
as 499.

### Further corrections

* **The ASR verifier needs acronym round-tripping too.** The pipeline feeds the
  model `see-dee-eff` so it spells CDF aloud, but Whisper transcribes what it
  hears as "CDF". Both sides must collapse to one token, or all 151 CDF chunks
  are false positives. Same class of bug as the digit issue, found by
  anticipating it rather than by another failed run.
* **The duration check must be affine, not linear.** `expected = len/cps`
  flagged "I signed." as a stall: 0.6 s predicted against 1.6 s actual, when
  1.6 s is entirely normal. Speech has fixed per-utterance overhead (onset,
  final consonant, trailing beat) independent of length, so the model is
  `0.55 + len/cps`. This produced 6 false positives on one English chapter and
  would have produced them on ~12% of every book. The corrected model still
  catches the real Chatterbox stall (5.1 s against 1.7 s expected).
* **Kokoro needs `ESPEAK_DATA_PATH` set explicitly.** Its bundled
  `espeakng-loader` ships a hard-coded CI path
  (`/Users/runner/work/...`) that does not exist, so G2P fails at first call.
  `brew install espeak-ng` plus the env var fixes it; the English G2P also
  needs the spaCy model `en_core_web_sm`.
* **Kokoro output is quiet, not clipped** (peaks 0.33-0.68 measured), the
  opposite of Silero. Mastering's RMS normalisation covers both.

## 6c. Final architecture: route by language

| language | backend | why |
|---|---|---|
| ru, uk, be | **Silero v5** | 0.0% CER, native stress/homographs, RTF 0.0066 |
| en, es, fr, it, pt, hi, ja, zh | **Kokoro-82M** | 0.0% CER, RTF 0.083, 12-48x cheaper than alternatives |
| anything else | Chatterbox Multilingual | 23 languages; the fallback, never the default |

Chatterbox is opt-in via `--backend chatterbox`, for when one cloned voice
across languages matters more than throughput. Silero and Kokoro coexist in the
main venv with Whisper (torch 2.13); only Chatterbox needs isolation, because
`chatterbox-tts` pins torch 2.6.

### Whole-library projection

Measured across the actual library — **79 EPUBs in ~/Downloads, all 79 parsed
with zero failures**, 66 Russian / 13 English, **53.1M characters = 983 hours
of audio**:

| stage | audio | measured RTF (incl. ASR QC) | wall clock |
|---|---|---|---|
| Silero (66 ru books) | 902.9 h | 0.062 | 56.0 h |
| Kokoro (13 en books) | 79.8 h | 0.108 | 8.6 h |
| **total** | **982.7 h** | | **64.6 h** |

The entire library converts in **under three days of unattended laptop time**,
for a few dollars of electricity. The same work on ElevenLabs Multilingual v3
at $100/M characters would cost **$5,307**.

The DGX Spark is therefore *not needed* for the library as it stands. It becomes
relevant only if the aesthetic judgement goes to Chatterbox for the English
shelf, in which case Turbo on the GB10 is the thing to benchmark.

## 6d. Russian stress — what listening found that measurement could not

Reported on listening: *«Мне русский не понравился. Ударения неправильные»*, and
specifically *«Why после́ not п́осле?»*, plus Silero "swallowing the о in words
like Дон".

**The verification harness is structurally blind to stress.** An ASR round trip
compares word identity, and Whisper transcribes за́мок and замо́к as the same
string "замок". So 0.0% CER established that the right *words* were spoken and
said nothing whatever about prosody. The earlier claim that Silero "correctly
disambiguated готов/готов" was unfounded — Whisper returns that text either way.
Stress is only detectable by ear, or by diffing against a dedicated
accentuation model. **Any future claim about Russian stress needs a listening
test, not a CER number.**

### Fix: mark stress explicitly rather than trusting the model

Silero's symbol set includes `+`, so stress can be supplied in the input.
`ru_stress.Accentuator` marks the text with RUAccent and the backend passes
`put_accent=False` so Silero cannot overrule it. Five defects had to be fixed
on top of the library:

1. **RUAccent's dictionary is wrong for «после».** Its 3.19M-entry dictionary
   maps `после` to `посл+е`, and the neural accentuator agrees. That is simply
   incorrect — the stress is always on the first syllable. `OVERRIDES` in
   `ru_stress.py` forces the correct form, and is the hook for any further word
   the library gets wrong.
2. **The dictionary is case-sensitive, so sentence-initial words were
   systematically degraded.** `после` hits the dictionary; `После` misses it
   entirely and falls through to the neural accentuator, which returned
   `Посл+е`. Worse, `Однако` came back with *no mark at all* while `однако`
   gives `одн+ако`. Since only capitalised-at-sentence-start is grammatical
   (mid-sentence capitals are proper nouns, where case can be lexically
   meaningful), only sentence-initial words are lowered for lookup.
3. **Lowering the first word perturbs the whole sentence.** Doing it in place
   changed the context the homograph model sees and measurably degraded *other*
   words in the same sentence (`сж+атый` lost its mark). So the accentuation now
   runs twice: the original text supplies every word's stress, and the lowered
   pass is consulted only for the sentence-initial position it was meant to fix.
4. **Single-vowel marks must be kept — this was the «Дон» defect.** An earlier
   version stripped them, on the theory that Silero's `stress_single_vowel`
   covers that case. It does not when `put_accent=False`. RUAccent correctly
   emits `Д+он`; stripping it left Silero with no stress information at all and
   the vowel came out reduced. Same for `пр+ав`, `зн+ал`.
5. **Clitics must NOT be marked.** Marking every monosyllable produced a
   staccato, over-emphatic reading (`+И к+ак р+аз в +это`). Russian
   prepositions, conjunctions and particles lean on the following word, so
   `CLITICS` is exempted while content monosyllables keep their stress:
   `И как р+аз в +это вр+емя в н+ос`.

Anything still unmarked with a vowel is sent to the neural accentuator as a
last resort, so no word reaches Silero without stress information.

Cost of all this: RTF 0.006 → 0.008–0.011. Negligible.

Measured over 12,454 words of the book, the corrected pipeline changes stress on
0.79% of words versus the naive single-pass version, 36% of those being words
that previously carried no mark at all.

### Chatterbox needs a completely different stress encoding

`+` marks **destroy** Chatterbox's output — it vocalises the plus signs:

| encoding | CER | transcript |
|---|---|---|
| plain (no marks) | 0.0% | И как раз в это время в нос корабля врезался метеорит. |
| `+` marks | **58.5%** | И как эрба асф, это в это воро емя, в это ба ос, карабал я воро езлса метеорорет. |
| **U+0301 combining acute** | **0.0%** | И как раз в это время в нос корабля врезался метеорит. |

The combining acute is the ordinary orthographic way to write Russian stress and
so appears in training data; `+` is a Silero-specific convention that nothing
else understands. `ru_stress.to_acute()` converts between them, and the
Chatterbox backend applies it automatically for Russian. ё takes no acute, being
inherently stressed.

This also means **CosyVoice 3 is a poor fit for this library** despite covering
Russian: like Chatterbox it has no explicit stress control, and Alibaba's own
model card flags Russian as needing debugging. A model that learns stress
statistically cannot be corrected when it gets a word wrong; Silero plus an
override dictionary can.

## 6e. Round two: ESpeech, and why Chatterbox cannot be fixed

Listening rejected Silero's stress and then, once stress was marked explicitly,
still preferred Chatterbox's *voice* ("Chatterbox is way way too better") while
reporting more stress errors from it (отды́ха, ноги́). That combination is the
whole problem, and resolving it needed a third model.

### Chatterbox receives stress marks but does not obey them

A controlled test: same seed, same sentence, the acute moved to each syllable of
«отдыха» in turn.

| input | output |
|---|---|
| `общения и отдыха` (unmarked) | 1.68 s |
| `о́тдыха` (correct) | 2.48 s |
| `отды́ха` (deliberately wrong) | 1.72 s |
| `отдыха́` (deliberately wrong) | 1.68 s |

Every marked variant differs from the unmarked one and from the others, so the
diacritic *is* reaching the model — but the audible stress stayed wrong. Three
independently reported words (после, отдыха, ноги) were marked correctly by the
pipeline and still mis-stressed by Chatterbox. **Chatterbox cannot be
stress-controlled through text.** For a Russian shelf that is disqualifying, and
no amount of notation work will change it.

### ESpeech-TTS-1: the combination that actually exists

Apache 2.0, F5/DiT architecture — the same family as the voice quality that was
preferred — and it consumes RUAccent `+` marks natively; its own reference
implementation is literally `if '+' in text: use as-is`. Measured on the M4 Max
via MPS, on the meteorite passage:

| model | CER | RTF | notes |
|---|---|---|---|
| **ESpeech-RLV2** | **0.7%** | 0.84 | best intelligibility of any Russian backend tested |
| ESpeech-SFT | 3.5% | 0.78 | SFT variant, weaker |
| ESpeech-RLV2 @ speed 0.82 | 0.7% | 1.02 | see pacing note |
| silero-aidar + RUAccent | 1.7% | 0.008 | stress fully correct, flattest voice |
| chatterbox-ru-acute | 0.0% | 3.75 | ignores the marks |

**Pacing caveat:** ESpeech at `speed=1.0` renders the passage in 75 s where
Silero takes 113 s — 21 chars/second against 14. That is fast for narration, so
a `speed=0.82` variant (89 s, ~18 chars/s) is rendered for comparison. Whichever
is chosen, it is a per-book constant, not a per-chunk risk.

At RTF 0.84 the Russian shelf is ~770 h on this Mac against Silero's 56 h. That
is the real trade: roughly 14x the compute for a materially better voice with
stress still under our control. **On the Spark this is the workload that
justifies the hardware** — and unlike Chatterbox, the quality is worth the GPU.

### CosyVoice 3: does not run on Apple Silicon

Evaluated on request. It never reached a Russian-quality judgement because it
fails to generate at all — and **English fails identically**, which rules out the
Russian-support caveat as the cause. The traceback lands in
`cosyvoice/llm/llm.py:479 inference`, the LLM emits almost no speech tokens, and
a downstream `CausalConv1d` then dies on a 3-frame input against a kernel of 4.
The run log is full of `torch.cuda.amp.autocast` warnings on a CPU-only process:
the codebase is written for CUDA and is not validated on MPS.

Getting there was also a fight worth recording, since the Spark will repeat it:

* the PyPI `cosyvoice` package is only an API client — the model needs the repo;
* `requirements.txt` pins `onnxruntime==1.18.0` against a CUDA-only extra index,
  so a plain install fails with a dependency-confusion error;
* `openai-whisper`'s build needs `pkg_resources`, i.e. `setuptools<81` — the same
  failure class as Chatterbox's perth watermarker;
* CosyVoice3 has its own class and its own `cosyvoice3.yaml`, and a different
  constructor signature from CosyVoice2 (no `load_jit`);
* `prompt_wav` is a **path**, not a tensor — the frontend calls `load_wav` itself;
* `torchaudio` 2.11 routes `load()` through torchcodec, whose dylib will not link
  against FFmpeg 9; both the ESpeech and CosyVoice runners shim audio I/O to
  soundfile.

**Verdict: retry CosyVoice 3 on the DGX Spark, where CUDA is available.** On the
Mac it is not a candidate. Note that even if it runs there, it shares
Chatterbox's fatal property for this library — no explicit stress control — so it
would have to be *audibly* right about Russian stress by luck, with no
correction mechanism when it is not.

## 6f. The word-drop hole in the verifier, and why retries matter

Reported by ear on the ESpeech-SFT sample: it swallowed «шестнадцать».
Transcription confirmed it rendered the word as «шить». RLV2 got it right at
both speeds; this is the F5 word-skipping failure mode, and SFT shows it more
(3.5% CER against RLV2's 0.7%).

**CER alone would not have caught it.** That chunk is 149 characters, so one
mangled word scores **5.4% CER** -- comfortably under the 14% Russian threshold.
Length normalisation hides short errors in long chunks, which is precisely the
class of error that matters most: a dropped numeral is a factual change to the
book ("sixteen passengers died" becoming "sew passengers died").

`verify.py` now adds two checks that are *not* length-normalised:

* **Numeral integrity** -- every spelled-out number in the source must appear in
  the transcript. Any missing numeral flags the chunk on its own.
* **Long-word integrity** -- content words of 6+ characters must survive; two or
  more missing flags the chunk.

Both match fuzzily (bounded edit distance ~20% of word length) so Russian case
endings and ASR spelling drift do not raise false alarms. Verified: the real
drop is caught, a digit-vs-word transcript ("16" for «шестнадцать») is not
flagged, and a transcript with three inflection differences is not flagged.

**The drop is stochastic, so retries fix it.** Re-rendering the same chunk at
five different seeds produced five different waveforms and «шестнадцать» came
out correct in all five. Confirmed end to end: chapter 1 through ESpeech-SFT
with verification on gave 74 chunks, RTF 0.97 including QC and retries, **zero
chunks flagged**, and the delivered audio contains «шестнадцать»,
«восемнадцать» and «двенадцать» intact.

The practical consequence: with ESpeech, **verification is not optional**.
`--no-verify` roughly doubles throughput and reintroduces exactly this defect.

## 6g. Pacing artifacts, and the shift to on-demand conversion

Reported by ear: an unnaturally long pause inside «Иоган Кеплер».

It was not the chunk splitter -- that chunk fitted in one F5 batch (380 bytes
against a 381-byte budget). It was the **guillemets**. They are present in
ESpeech's vocabulary, so not unknown tokens; the model has simply learned that
«» marks dialogue and pauses there. Measured on one sentence:

| variant | longest internal gap |
|---|---|
| `«Иоган Кеплер»` | **0.83 s** |
| `Иоган Кеплер` | **0.37 s** |
| `"Иоган Кеплер"` | 0.62 s |
| `Иоган Кеплер,` | 0.93 s |

Quotes around actual speech *should* pause, so this cannot be a blanket strip.
Measured across 25 Russian books in the library: 15,747 quoted spans, **63%
short name/title spans** («Найти идею», «Альпина Бизнес Букс», «Легенды о
звездных капитанах») and 37% speech-like. `normalise.strip_title_quotes`
therefore drops the marks only around spans of at most 5 words with no sentence
punctuation inside. Verified after the fix: the gap at the ship name fell to
0.27 s while the deliberate paragraph pauses were untouched.

**Generic backstop.** Rather than diagnose every such artifact,
`master.clamp_internal_silence` now caps dead air *inside* a chunk at 450 ms.
Pacing is supposed to come from the pauses the mastering stage inserts
deliberately, so anything longer inside a chunk is trimmed. It runs per chunk
*before* the deliberate pauses are added, so a 1.3 s bogus gap becomes 0.45 s
while the 700 ms paragraph pauses survive. On the Knyazev passage it clamped 15
gaps. The count is reported in the chapter stats, so a book that needs a lot of
clamping is visible rather than silently repaired.

### Scope change: convert on demand, not in bulk

The library will be converted a book at a time as wanted, not as one 66-book
sweep. That retires the whole-shelf projections as a decision input and changes
what matters:

* **Per-book wall clock is the only figure that counts.** ESpeech + a Knyazev
  reference runs at RTF ~0.9-1.1 including verification, so an average 13.7-hour
  book is roughly **12-15 hours** -- one unattended overnight run per book.
* **The DGX Spark is no longer needed.** It was only justified by the 33-to-41
  day shelf estimate. For one book overnight, the Mac is sufficient.
* **Quality now beats throughput outright.** With no bulk deadline there is no
  reason to prefer Silero's 51-minutes-per-book over the voice that is actually
  wanted.
* **Resumability matters more than speed.** Chapter-level caching already exists
  (`--force` to re-render), so an interrupted overnight run resumes rather than
  restarting.

## 7. Phased plan

**Phase 0 — Bake-off (half a day, highest value).** Take one chapter of the
Harrison book plus one English chapter. Generate with (a) Chatterbox Multilingual,
(b) Chatterbox + RUAccent stress marks, (c) Silero v5_5_ru. Listen to all three.
Measure RTF and find the concurrency plateau on the Spark. **Decide the Russian
backend on evidence before building anything.**

**Phase 1 — Skeleton, one book end to end.** Ingest → normalise → synthesise →
master → package, no QC loop, no parallelism. Goal is a playable M4B in Books with
correct chapters, cover, and `stik=2`. Accept bad chunks for now.

**Phase 2 — Reliability.** Add the SQLite manifest, resumability, the ASR
verification loop, retry-with-new-seed, and the mechanical outlier detectors.
This is what makes unattended overnight runs trustworthy.

**Phase 3 — Throughput.** N concurrent Spark workers, MBP verifying in parallel,
a work queue over the library, `rsync` or a share between the boxes.

**Phase 4 — Polish as needed.** Per-book pronunciation dictionaries, better TOC
flattening heuristics, separate voices for narrator vs dialogue (genuinely hard —
treat as optional).

---

## 8. Suggested layout

```
ebooker/
  DESIGN.md
  pyproject.toml            # uv-managed
  ebooker/
    ingest.py               # EPUB → book.json (spine, NCX, cover, language)
    normalise/
      __init__.py
      common.py             # sentence segmentation, chunk packing
      ru.py                 # case-aware numbers, ё, RUAccent, homographs
      en.py
    synth/
      chatterbox.py         # the backend
      silero.py             # RU fallback from Phase 0
    verify.py               # whisper.cpp round-trip + CER + outlier detectors
    master.py               # pauses, ACX loudness, chapter concat
    package.py              # M4B + chapters + cover + stik
    manifest.py             # SQLite chunk state
    cli.py
  voices/
    narrator_ru.wav         # 10-20 s clean reference
    narrator_en.wav
  work/<book-slug>/         # chunks, manifest.db, chapter wavs
  out/                      # finished .m4b
```

---

## 9. Risks, honestly

| Risk | Severity | Mitigation |
|---|---|---|
| Chatterbox Russian prosody/stress not good enough | **High** — it's the whole premise | Phase 0 bake-off; Silero v5_5_ru as fallback |
| Spark aarch64/CUDA-13 environment burns a day+ | Medium | Community wheels, NGC containers, cu130 index |
| Apple Books won't sync to phone | Medium — already confirmed true | Emit standard M4B; Audiobookshelf as alternative front end |
| Per-book wall-clock worse than hoped | Medium | Measure in Phase 0 before planning the library run |
| Whisper QC false-positives on Russian | Low | Separate per-language CER thresholds; review queue not auto-discard |
| FB2-origin EPUBs parse inconsistently | Low | Defensive ingest; `needs_review` on structural anomalies |

## 10. Licensing

Chatterbox is **MIT** — no restriction on use. Silero's main models are
**CC-BY-NC**, with MIT-licensed base CIS models; personal library conversion is
fine under either. Note Chatterbox embeds a **Perth audio watermark** in its
output; irrelevant for personal listening, worth knowing it's there.

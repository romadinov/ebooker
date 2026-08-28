"""Mechanical QC for synthesised audio.

Autoregressive TTS drops and hallucinates words silently, and a 13.6-hour book
is ~5,500 chunks -- far too many to audition. So every chunk is checked two ways:

* cheap signal checks (silence, clipping, duration outliers, tail repetition)
  which need no model and catch the gross failures;
* an ASR round trip, which catches the subtle ones -- wrong or missing words in
  otherwise healthy-sounding audio.

A chunk that fails is requeued for synthesis with a different seed, not
discarded, and only lands in `needs_review` after the retry budget is spent.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

# Chars-per-second of speech, used for the duration sanity check. Calibrated
# per book at runtime; this is only the cold-start prior.
DEFAULT_CPS = 15.0

# Fixed per-utterance overhead, in seconds. Speech is affine in length, not
# linear: onset, the final consonant and a natural trailing beat cost roughly
# the same whatever the sentence length. A purely linear model expects "I
# signed." to take 0.6s when 1.6s is entirely normal, which made every very
# short chunk a false positive -- and 12% of chunks are under 40 chars.
UTTERANCE_OVERHEAD_S = 0.55


@dataclass
class Checks:
    ok: bool
    cer: float | None = None
    transcript: str = ""
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


# --------------------------------------------------------------------------
# text normalisation for fair comparison
# --------------------------------------------------------------------------

# Russian "one" and "two" are gendered, and num2words only produces the
# masculine. So a source that says "две" (feminine) against an ASR that writes
# "2" canonicalises to "две" versus "два" and reads as a dropped numeral. That
# single mismatch produced 146 of 424 flags on one book -- a third of the whole
# review list. Gendered forms fold to one representative on both sides.
_RU_GENDER = {
    "одна": "один", "одно": "один", "одну": "один", "одной": "один",
    "две": "два", "двух": "два", "двум": "два", "двумя": "два",
    "первая": "первый", "первое": "первый", "вторая": "второй",
    "втором": "второй", "второе": "второй",
}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

# A British voice makes the ASR transcribe in British spelling, so the written
# American form is reported missing: "donuts" against "doughnuts", "Center"
# against "Centre". Both spellings fold to one form before comparison.
_SPELLING = {
    "centre": "center", "theatre": "theater", "metre": "meter",
    "litre": "liter", "fibre": "fiber", "colour": "color",
    "favour": "favor", "honour": "honor", "labour": "labor",
    "neighbour": "neighbor", "behaviour": "behavior", "flavour": "flavor",
    "rumour": "rumor", "humour": "humor", "armour": "armor",
    "doughnut": "donut", "doughnuts": "donuts", "grey": "gray",
    "realise": "realize", "realised": "realized", "recognise": "recognize",
    "organisation": "organization", "apologise": "apologize",
    "aeroplane": "airplane", "kerb": "curb", "storey": "story",
    "practise": "practice", "defence": "defense", "offence": "offense",
    "travelled": "traveled", "travelling": "traveling", "cancelled": "canceled",
}


def canon(s: str, lang: str = "ru") -> str:
    """Collapse both texts to a form where only real word errors differ.

    ASR will not reproduce the source's punctuation, casing, or ё/е choice, and
    penalising those would drown out the errors that matter.

    Digits matter especially: the pipeline feeds the model spelled-out numerals
    ("девяносто два"), but Whisper transcribes them back as digits ("92"). Left
    alone that inflates CER on every chunk containing a number, so digits in
    either side are expanded to words before comparison.
    """
    s = unicodedata.normalize("NFC", s).casefold()
    if lang == "ru":
        s = s.replace("ё", "е")
    if lang != "ru":
        # Before hyphens are stripped, since the spelled form uses them.
        s = _collapse_spelled_acronyms(s)
    # Em/en dashes are punctuation and become spaces. ASCII hyphens JOIN, they
    # do not separate: English writes "grand-kids", "good-bye", "seventy-fifth"
    # while the ASR returns "grandkids", "goodbye", "75th". Splitting on the
    # hyphen manufactured a word-count deficit on every such compound.
    s = s.replace("—", " ").replace("–", " ").replace("-", "")
    s = _digits_to_words(s, lang)
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    if s:
        if lang == "ru":
            s = " ".join(_RU_GENDER.get(w, w) for w in s.split())
        else:
            s = " ".join(_SPELLING.get(w, w) for w in s.split())
    return s


# Reverse of normalise._EN_LETTER_SAY, for undoing spelled-out acronyms before
# comparison. The pipeline feeds the model "see-dee-eff" so it pronounces CDF
# letter by letter, but Whisper transcribes what it hears as "CDF". Both sides
# must collapse to the same token or every acronym chunk is a false positive --
# and CDF alone appears 151 times in one book.
_LETTER_SAY_REV = {
    "ay": "a", "bee": "b", "see": "c", "dee": "d", "ee": "e", "eff": "f",
    "gee": "g", "aitch": "h", "eye": "i", "jay": "j", "kay": "k", "el": "l",
    "em": "m", "en": "n", "oh": "o", "pee": "p", "cue": "q", "ar": "r",
    "ess": "s", "tee": "t", "you": "u", "vee": "v", "double-you": "w",
    "ex": "x", "why": "y", "zee": "z", "zed": "z",
}
_SPELLED_RUN = re.compile(
    r"\b(?:" + "|".join(sorted(_LETTER_SAY_REV, key=len, reverse=True)) + r")"
    r"(?:[-\s](?:" + "|".join(sorted(_LETTER_SAY_REV, key=len, reverse=True)) + r"))+\b")


def _collapse_spelled_acronyms(s: str) -> str:
    """"see-dee-eff" and "C D F" both become "cdf"."""
    def join(m: re.Match) -> str:
        parts = re.split(r"[-\s]+", m.group(0))
        return "".join(_LETTER_SAY_REV.get(p, p) for p in parts)

    s = _SPELLED_RUN.sub(join, s)
    # Whisper may also emit "C.D.F." or "C D F".
    s = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", ""), s)
    s = re.sub(r"\b(?:[a-z] ){1,4}[a-z]\b",
               lambda m: m.group(0).replace(" ", "")
               if len(m.group(0).replace(" ", "")) >= 2 else m.group(0), s)
    return s


def _digits_to_words(s: str, lang: str) -> str:
    """Expand digits so a spelled-out source matches a digit-writing ASR.

    Decimals must be handled before integers: Whisper renders "две целых восемь
    десятых" as "2,8", and expanding the 2 and 8 separately yields "два восемь",
    which reads as a 33% error on audio that was in fact perfect.
    """
    if not any(ch.isdigit() for ch in s):
        return s
    from num2words import num2words

    code = "ru" if lang == "ru" else "en"

    def one(txt: str, is_float: bool) -> str:
        try:
            return num2words(float(txt) if is_float else int(txt), lang=code)
        except (NotImplementedError, OverflowError, ValueError):
            return txt

    # Military time first, matching what normalise.expand_english produces --
    # otherwise "0600" canonicalises to "six hundred" and loses the leading
    # "oh" that the pipeline actually fed the model.
    def miltime(m: re.Match) -> str:
        h, mn = int(m.group(0)[:2]), int(m.group(0)[2:])
        hh = ("oh " + one(str(h), False)) if h < 10 else one(str(h), False)
        return f"{hh} hundred" if mn == 0 else f"{hh} {one(str(mn), False)}"

    # Thousands separators: "2,000" hit the decimal rule and became "two",
    # losing "thousand" entirely and reporting it as an unspoken numeral. The
    # pattern requires exactly three trailing digits, so Russian decimals
    # written with a comma ("3,2") are untouched.
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)

    if code == "en":
        # Clock times first: the ASR writes "8.30am" or "8:30 a.m." where the
        # pipeline fed the model "eight thirty in the morning". Left to the
        # decimal rule, "8.30" would become "eight point three".
        def clock(m: re.Match) -> str:
            from .normalise import clock_words
            return clock_words(int(m.group(1)), int(m.group(2)), m.group(3) or "")

        s = re.sub(r"\b(\d{1,2})[:.]([0-5]\d)\s*([ap]\.?m\.?)", clock, s)
        s = re.sub(r"\b(\d{1,2}):([0-5]\d)()\b", clock, s)
        s = re.sub(r"\b0[0-9][0-5][0-9]\b", miltime, s)

        def ordinal(m: re.Match) -> str:
            try:
                return num2words(int(m.group(1)), lang="en", to="ordinal")
            except (NotImplementedError, OverflowError, ValueError):
                return m.group(0)

        # "75th" -> "seventy-fifth", so it matches a source that spelled it out.
        s = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", ordinal, s, flags=re.IGNORECASE)
    # "30-го", "1-й", "2-е": drop the ordinal suffix so the number expands and
    # stem matching can pair it with the written ordinal ("тридцатого").
    s = re.sub(r"\b(\d+)-(?:го|му|м|й|я|е|ю|ой|ые|ых|ом)\b", r"\1", s)
    s = re.sub(r"\d+[.,]\d+",
               lambda m: one(m.group(0).replace(",", "."), True), s)
    return re.sub(r"\d+", lambda m: one(m.group(0), False), s)


def cer(reference: str, hypothesis: str, lang: str = "ru") -> float:
    """Character error rate over canonicalised text, via Levenshtein distance."""
    a, b = canon(reference, lang), canon(hypothesis, lang)
    if not a:
        return 0.0 if not b else 1.0
    # Two-row DP; chunks are short so this is cheap.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[len(b)] / len(a)


# --------------------------------------------------------------------------
# word-level integrity
# --------------------------------------------------------------------------

# Numerals, spelled out. A dropped number is a factual error in a narration --
# "sixteen passengers died" becoming "sew passengers died" -- and it is exactly
# the case CER hides, because one short word inside a long chunk barely moves a
# length-normalised score. Measured: ESpeech-SFT rendered "шестнадцать" as
# "шить" for a chunk CER of 5.4%, well under the 14% Russian threshold.
_RU_NUM_WORDS = set("""
ноль один одна одно два две три четыре пять шесть семь восемь девять десять
одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать
семнадцать восемнадцать девятнадцать двадцать тридцать сорок пятьдесят
шестьдесят семьдесят восемьдесят девяносто сто двести триста четыреста
пятьсот шестьсот семьсот восемьсот девятьсот тысяча тысячи тысяч миллион
миллиона миллионов миллиард целых десятых сотых первый второй третий
""".split())
_EN_NUM_WORDS = set("""
zero one two three four five six seven eight nine ten eleven twelve thirteen
fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty
sixty seventy eighty ninety hundred thousand million billion point first second
third fourth fifth
""".split())

LONG_WORD = 5           # content words at least this long must survive


def _stem(w: str) -> str:
    """Crude stem for matching across inflection.

    Russian case endings differ by two or three characters, which defeats a
    tight edit-distance budget: "внешняя" against ASR's "внешне" is distance 2,
    "похожего" against "похожа" is 3. Both words were spoken. Comparing stems
    instead avoids flagging inflection as a dropped word, while still catching
    real substitutions -- "шестнадцать" stems to "шестна", which "шить" does
    not match.
    """
    return w[: min(6, max(4, len(w) - 3))]


def _close(a: str, b: str) -> bool:
    """Fuzzy word match, tolerant of inflection and ASR spelling drift."""
    if a == b:
        return True
    # Same stem is treated as the same word -- but only when the two are of
    # comparable length. Allowing a much shorter hypothesis word to satisfy a
    # longer written one is how "ракет" was accepted for "ракетного" while the
    # audio actually said "ракет Монибуса".
    if len(a) >= 5 and len(b) >= 4 and abs(len(a) - len(b)) <= 3:
        if b.startswith(_stem(a)) or a.startswith(_stem(b)):
            return True
    if abs(len(a) - len(b)) > max(2, len(a) // 3):
        return False
    budget = max(1, len(a) // 4)
    # cheap bounded edit distance
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > budget:
            return False
        prev = cur
    return prev[len(b)] <= budget


def missing_words(reference: str, hypothesis: str, lang: str = "ru") -> tuple[list[str], list[str]]:
    """Source words absent from the transcript: (numerals, other long words).

    Numerals are matched strictly; other words fuzzily, so Russian case endings
    and ASR spelling variation do not raise false alarms.
    """
    nums = _RU_NUM_WORDS if lang == "ru" else _EN_NUM_WORDS
    hyp = canon(hypothesis, lang).split()
    hyp_set = set(hyp)

    # Hyphens are joined by canon, which turns "brat-and-beer-filled" into
    # "bratandbeerfilled" -- a token no transcript can ever contain. Ref tokens
    # are therefore built per whitespace-separated word, keeping the parts, so a
    # compound counts as spoken if either the joined form or every part is there.
    ref_pairs: list[tuple[str, list[str]]] = []
    for raw in re.split(r"\s+", reference):
        whole = canon(raw, lang)
        if not whole:
            continue
        parts = [c for c in (canon(x, lang) for x in raw.split("-")) if c]
        for w in whole.split():
            ref_pairs.append((w, parts if len(parts) > 1 else []))
    ref = [w for w, _ in ref_pairs]

    # The ASR splits compounds the source writes solid ("anytime" -> "any
    # time"), so a joined form of the hypothesis is checked as well.
    joined = "".join(hyp)

    def present(w: str) -> bool:
        if w in hyp_set or any(_close(w, h) for h in hyp):
            return True
        # The ASR both splits solid compounds and joins spaced ones: the source
        # "skip drive" came back as "skip-drive", which the hyphen-join turns
        # into one token, hiding "drive".
        return len(w) >= 5 and w in joined

    def spoken(w: str, parts: list[str]) -> bool:
        if present(w):
            return True
        # A compound counts as spoken when all of its parts are.
        return bool(parts) and all(present(p) or len(p) < 3 for p in parts)

    missing_nums, missing_long = [], []
    for w, parts in ref_pairs:
        if w in nums:
            if not spoken(w, parts):
                missing_nums.append(w)
        elif len(w) >= LONG_WORD:
            if not spoken(w, parts):
                missing_long.append(w)
    return missing_nums, missing_long


# --------------------------------------------------------------------------
# signal-level checks (no model needed)
# --------------------------------------------------------------------------

def signal_checks(
    audio: np.ndarray, sample_rate: int, text: str,
    cps: float = DEFAULT_CPS, tolerance: float = 2.2,
) -> list[str]:
    reasons: list[str] = []
    if audio.size == 0:
        return ["empty audio"]

    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(np.square(audio))))
    dur = audio.size / sample_rate

    if peak < 0.02:
        reasons.append(f"near-silent (peak {peak:.3f})")
    if peak >= 1.0:
        reasons.append(f"clipped (peak {peak:.2f})")
    if rms < 0.005:
        reasons.append(f"very low level (rms {rms:.4f})")

    expected = UTTERANCE_OVERHEAD_S + len(text) / cps
    # Short utterances are dominated by the fixed overhead, so the prediction is
    # weakest exactly where the window is tightest. Widen it there: a one-word
    # line can legitimately run at half the predicted length.
    tol = tolerance * (1.6 if len(text) < 25 else 1.25 if len(text) < 60 else 1.0)
    if dur > expected * tol:
        reasons.append(f"too long: {dur:.1f}s vs ~{expected:.1f}s expected (stall or loop?)")
    if dur < expected / tol:
        reasons.append(f"too short: {dur:.1f}s vs ~{expected:.1f}s expected (truncated?)")

    if _tail_loop(audio, sample_rate):
        reasons.append("repetition loop detected in tail")
    return reasons


def _tail_loop(audio: np.ndarray, sr: int, window: float = 1.0) -> bool:
    """Detect a stuck generation by autocorrelating the final seconds.

    A looping model repeats near-identical audio at a short lag, which shows up
    as an autocorrelation peak far above the local baseline.
    """
    n = int(window * sr)
    if audio.size < 4 * n:
        return False
    tail = audio[-2 * n:].astype(np.float64)
    tail = tail - tail.mean()
    if not np.any(tail):
        return False
    ac = np.correlate(tail, tail, mode="full")[len(tail) - 1:]
    ac /= ac[0] or 1.0
    lo, hi = int(0.08 * sr), min(int(0.9 * sr), len(ac) - 1)
    if hi <= lo:
        return False
    seg = ac[lo:hi]
    return bool(seg.max() > 0.82 and seg.max() > 6 * np.median(np.abs(seg)))


# --------------------------------------------------------------------------
# ASR round trip
# --------------------------------------------------------------------------

class Transcriber:
    """Whisper wrapper with three backends, chosen by what the machine has.

    * **mlx** -- mlx-whisper on the Apple Silicon GPU.
    * **openai** -- openai-whisper through torch, which is the only one that
      reaches CUDA on aarch64.
    * **faster** -- faster-whisper (CTranslate2) on CPU, the fallback.

    Backend choice matters for throughput far more than expected. Measured on
    a GB10 with a 7.2 s Russian clip:

        openai-whisper turbo    on CUDA   RTF 0.061
        openai-whisper large-v3 on CUDA   RTF 0.225
        faster-whisper small    on CPU    RTF 1.10
        faster-whisper large-v3 on CPU    RTF 4.38   <- unusable

    The aarch64 CTranslate2 wheel has no CUDA support and its CPU path is very
    slow, so on that machine ASR has to go through torch. Note also that
    distil-large-v3 is English-only: given Russian audio it returns an English
    translation, which would silently wreck the verification.

    All defaults stay in the large-v3-turbo family so CER thresholds calibrated
    on one machine hold on the other.
    """

    def __init__(self, model: str | None = None, backend: str | None = None,
                 device: str | None = None):
        self.backend = backend or _pick_asr_backend()
        if model is None:
            model = {"mlx": "mlx-community/whisper-large-v3-turbo",
                     "openai": "turbo",
                     "faster": "small"}[self.backend]
        self.model = model
        self.device = device
        self._impl = None

    def _load(self):
        if self._impl is not None:
            return self._impl
        if self.backend == "mlx":
            import mlx_whisper
            self._impl = mlx_whisper
        elif self.backend == "openai":
            import whisper, torch
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._impl = whisper.load_model(self.model, device=dev)
            self.device = dev
        else:
            import os
            from faster_whisper import WhisperModel

            def build(dev: str):
                compute = "float16" if dev == "cuda" else "int8"
                # CPU inference wants the cores; on the Spark that is 20 of
                # them, and running ASR on CPU beside GPU synthesis is the
                # better split anyway -- neither waits on the other.
                kw = {} if dev == "cuda" else {"cpu_threads": max(1, (os.cpu_count() or 4) - 2)}
                return WhisperModel(self.model, device=dev, compute_type=compute, **kw)

            order = [self.device] if self.device else []
            if not order:
                try:
                    import torch
                    order = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
                except Exception:
                    order = ["cpu"]
            if "cpu" not in order:
                order.append("cpu")
            last = None
            for dev in order:
                try:
                    self._impl = build(dev)
                    self.device = dev
                    break
                except Exception as e:
                    # The aarch64 CTranslate2 wheel is built without CUDA, so
                    # asking for it raises rather than degrading. Fall through.
                    last = e
            if self._impl is None:
                raise RuntimeError(f"could not start faster-whisper: {last}")
        return self._impl

    def __call__(self, audio: np.ndarray, sample_rate: int, lang: str) -> str:
        impl = self._load()
        if sample_rate != 16000:
            audio = _resample(audio, sample_rate, 16000)
        audio = audio.astype(np.float32)
        if self.backend == "mlx":
            r = impl.transcribe(
                audio, path_or_hf_repo=self.model, language=lang,
                temperature=0.0, condition_on_previous_text=False,
                fp16=True, verbose=None)
            return (r.get("text") or "").strip()
        if self.backend == "openai":
            r = impl.transcribe(
                audio, language=lang, temperature=0.0,
                condition_on_previous_text=False,
                fp16=(self.device == "cuda"), verbose=None)
            return (r.get("text") or "").strip()
        segments, _ = impl.transcribe(
            audio, language=lang, temperature=0.0,
            condition_on_previous_text=False, beam_size=1, vad_filter=False)
        return " ".join(seg.text for seg in segments).strip()


def _pick_asr_backend() -> str:
    try:
        import mlx_whisper  # noqa: F401
        return "mlx"
    except Exception:
        pass
    try:
        import torch, whisper  # noqa: F401
        if torch.cuda.is_available():
            return "openai"      # the only route to CUDA on aarch64
    except Exception:
        pass
    return "faster"


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear resample. Adequate for ASR input; not used on delivered audio."""
    if src == dst:
        return x
    n = int(round(x.size * dst / src))
    return np.interp(np.linspace(0, x.size - 1, n),
                     np.arange(x.size), x).astype(np.float32)


def classify(attempts: list[Checks], spread: float = 0.06) -> str:
    """Separate a model failure from text that ASR simply cannot score.

    Generation is stochastic, so a real synthesis defect usually clears -- or at
    least moves -- when the seed changes. A CER that stays high *and* stays
    nearly identical across seeds is evidence about the text, not the audio:
    transliterated or invented words (the corpus has Nahuatl and Otomi
    epigraphs) transcribe badly no matter how cleanly they are spoken.

    Such chunks go to human review rather than burning more GPU on retries.
    """
    scored = [a.cer for a in attempts if a.cer is not None]
    if any(a.ok for a in attempts):
        return "ok"
    if len(scored) >= 2 and (max(scored) - min(scored)) < spread:
        return "unverifiable_text"
    return "synthesis_failure"


def check(
    audio: np.ndarray, sample_rate: int, text: str, lang: str,
    transcriber: Transcriber | None = None,
    cer_threshold: float | None = None,
    cps: float = DEFAULT_CPS,
) -> Checks:
    """Full check. Russian ASR is noisier than English, so thresholds differ."""
    if cer_threshold is None:
        cer_threshold = 0.14 if lang == "ru" else 0.08
        # A one-second clip gives the ASR little to work with, so the same bar
        # that is right for a paragraph produces false alarms on a two-word
        # reply. Loosen it in proportion to how little text there is.
        if len(text) < 30:
            cer_threshold *= 2.2
        elif len(text) < 70:
            cer_threshold *= 1.5

    reasons = signal_checks(audio, sample_rate, text, cps=cps)
    score = None
    transcript = ""
    if transcriber is not None:
        transcript = transcriber(audio, sample_rate, lang)
        score = cer(text, transcript, lang)
        if score > cer_threshold:
            reasons.append(f"CER {score:.1%} over {cer_threshold:.0%} threshold")
        # Not length-normalised, so a single mangled word in a long chunk is
        # still caught. Any lost numeral is fatal on its own.
        lost_nums, lost_long = missing_words(text, transcript, lang)
        if lost_nums:
            reasons.append(f"numeral(s) not spoken: {', '.join(lost_nums)}")
        # One missing word is enough when it is long: ASR rarely loses an
        # 8+ character word outright, so its absence is real. Shorter words
        # need corroboration to avoid flagging ASR noise.
        very_long = [w for w in lost_long if len(w) >= 7]
        if very_long:
            reasons.append(f"word(s) not spoken: {', '.join(very_long[:4])}")
        elif len(lost_long) >= 2:
            reasons.append(f"{len(lost_long)} long word(s) not spoken: "
                           f"{', '.join(lost_long[:4])}")
        # Short words carry meaning too -- "борт" is four characters and was
        # dropped without tripping any per-word check. A word-count deficit
        # catches those without matching each one: ASR merges and splits words
        # routinely, so the threshold is two rather than one.
        # Short words carry meaning too -- "борт" is four characters and was
        # dropped without tripping any per-word check, because the model merged
        # it into the next word ("на борт ракетного" -> "на бракетного"). That
        # leaves a word deficit of only 1 and a CER of 5.6%, which is *lower*
        # than a legitimately clean render whose inflections differ from the
        # ASR's (7.5% measured). Neither signal separates them alone; together
        # they do, because clean renders lose no words at all.
        n_ref = len(canon(text, lang).split())
        n_hyp = len(canon(transcript, lang).split())
        deficit = n_ref - n_hyp
        # On a very short chunk both statistics are unreliable: a 3-word line
        # of dialogue is about a second of audio, where the ASR frequently
        # drops the dialogue dash or merges a word, giving deficit 1 and a CER
        # of 15% on audio that is fine. Measured: every short-chunk flag in
        # chapters 21-25 was of this kind. So the deficit-of-one rule needs
        # enough words behind it to mean something.
        DEFICIT_MIN_WORDS = 12
        if deficit >= 2 and n_ref >= 6:
            reasons.append(f"{deficit} fewer words spoken than written "
                           f"({n_hyp} of {n_ref})")
        elif (deficit >= 1 and n_ref >= DEFICIT_MIN_WORDS
                and score is not None and score > 0.005):
            # A clean render loses no words at all: every verified-clean case
            # measured here has deficit 0, including ones with several
            # inflection differences. So any deficit with a non-zero CER is
            # worth a retry -- retries are cheap, a wrong word in a book is not.
            reasons.append(f"a word appears to be missing ({n_hyp} of {n_ref} "
                           f"words, CER {score:.1%})")
    # Clipping is repaired in mastering, not a reason to regenerate.
    fatal = [r for r in reasons if not r.startswith("clipped")]
    return Checks(ok=not fatal, cer=score, transcript=transcript, reasons=reasons)

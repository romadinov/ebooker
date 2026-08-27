"""Text -> narratable, chunked text.

Two jobs, in order:

1. Expand anything a TTS model would mispronounce (digits, designators,
   abbreviations) into plain words.
2. Segment into sentences and pack them into chunks small enough that the
   model stays coherent, never splitting mid-sentence.

Scope note, measured rather than assumed: the sample 738k-char Russian novel
contains only 156 numeric tokens (2.1 per 10k chars) and no Latin runs at all.
Number handling therefore stays deliberately pragmatic -- correct for the
common cases, and flagged for review rather than silently guessed otherwise.
Full Russian case agreement would need a morphological analyser (pymorphy3);
the hook for that is `_inflect`, currently a documented no-op.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from num2words import num2words

# Chatterbox stays coherent well below its ~30s ceiling; ~250 chars of Russian
# is roughly 17s of speech.
MAX_CHUNK_CHARS = 250
MIN_CHUNK_CHARS = 40

RU_LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

# Cyrillic letters read out individually inside designators (отсек 64А).
RU_LETTER_NAMES = {
    "а": "а", "б": "бэ", "в": "вэ", "г": "гэ", "д": "дэ", "е": "е", "ж": "жэ",
    "з": "зэ", "и": "и", "к": "ка", "л": "эль", "м": "эм", "н": "эн", "о": "о",
    "п": "пэ", "р": "эр", "с": "эс", "т": "тэ", "у": "у", "ф": "эф", "х": "ха",
    "ц": "цэ", "ч": "че", "ш": "ша", "щ": "ща", "э": "э", "ю": "ю", "я": "я",
}
EN_LETTER_NAMES = {c: c for c in "abcdefghijklmnopqrstuvwxyz"}

RU_ABBREV = {
    "т.е.": "то есть", "т. е.": "то есть",
    "т.д.": "так далее", "т. д.": "так далее",
    "т.п.": "тому подобное", "т. п.": "тому подобное",
    "т.к.": "так как", "т. к.": "так как",
    "и.о.": "исполняющий обязанности",
    "др.": "другие", "напр.": "например", "см.": "смотри",
    "стр.": "страница", "г.": "год", "гг.": "годы", "в.": "век", "вв.": "века",
    "тыс.": "тысяч", "млн.": "миллионов", "млрд.": "миллиардов",
    "руб.": "рублей", "коп.": "копеек",
}
EN_ABBREV = {
    "e.g.": "for example", "i.e.": "that is", "etc.": "and so on",
    "vs.": "versus", "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miz",
    "Dr.": "Doctor", "St.": "Saint", "Prof.": "Professor", "Jr.": "Junior",
    "Sr.": "Senior", "approx.": "approximately", "cf.": "compare",
}

# Tokens that end in '.' but do not end a sentence.
RU_NON_TERMINAL = {"т", "е", "д", "п", "к", "г", "гг", "в", "вв", "стр", "др",
                   "напр", "см", "тыс", "млн", "млрд", "руб", "коп", "им", "ул"}
EN_NON_TERMINAL = {"mr", "mrs", "ms", "dr", "st", "prof", "jr", "sr", "vs",
                   "etc", "eg", "ie", "no", "vol", "fig", "approx", "cf", "inc", "ltd"}

UNITS = {
    "ru": {"градус": ("градус", "градуса", "градусов"),
           "метр": ("метр", "метра", "метров"),
           "год": ("год", "года", "лет"),
           "лет": ("год", "года", "лет"),
           "день": ("день", "дня", "дней"),
           "час": ("час", "часа", "часов")},
}


# --------------------------------------------------------------------------
# pronunciation respellings
# --------------------------------------------------------------------------

# Transliterated foreign proper nouns break these models. Measured on ESpeech,
# the same sentence with the ship's name spelled six ways:
#   "Иоган"     -> "и ганг Кеплера"     (the word is lost)
#   "Иоганн"    -> "Эган Кеплер"
#   "И+о-г+ан"  -> "и Уганн-Кеплер"
#   "Йоган"     -> "Йоганн Кеплер"      correct
# The initial "Ио" is the problem; "Йо" is unambiguous. Respelling is the only
# lever that works -- stress marks and punctuation do not help, because the
# failure is in how the grapheme sequence is read, not where the stress falls.
#
# Names are book-specific, so this lives in an editable file rather than in code.
PRONUNCIATION_FILE = "pronunciation.txt"

_pron_cache: dict[str, dict[str, str]] = {}


def load_pronunciations(path: str | None = None) -> dict[str, str]:
    """Read respellings from a file: `written = respelled`, one per line."""
    import pathlib as _pl

    key = path or PRONUNCIATION_FILE
    if key in _pron_cache:
        return _pron_cache[key]
    out: dict[str, str] = {}
    p = _pl.Path(key)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            written, _, said = line.partition("=")
            written, said = written.strip(), said.strip()
            if written and said:
                out[written] = said
    _pron_cache[key] = out
    return out


def apply_pronunciations(text: str, table: dict[str, str] | None = None) -> tuple[str, int]:
    """Respell words the model reads wrong. Whole-word, case-insensitive."""
    table = load_pronunciations() if table is None else table
    if not table:
        return text, 0
    n = 0
    for written in sorted(table, key=len, reverse=True):
        pattern = re.compile(rf"(?<![^\W\d_]){re.escape(written)}(?![^\W\d_])",
                             re.IGNORECASE)
        said = table[written]

        def sub(m: re.Match) -> str:
            nonlocal n
            n += 1
            src = m.group(0)
            return said.capitalize() if src[:1].isupper() else said

        text = pattern.sub(sub, text)
    return text, n


# --------------------------------------------------------------------------
# quotation marks
# --------------------------------------------------------------------------

# Models treat «» as a dialogue boundary and insert a pause. Measured on
# ESpeech: «Иоган Кеплер» produced a 0.83 s internal gap where the same phrase
# without quotes gave 0.37 s -- audibly wrong for a ship's name mid-sentence.
#
# But quotes around actual speech *should* pause. Measured across 25 Russian
# books in this library: 15,747 quoted spans, 63% of them short name/title
# spans («Найти идею», «Альпина Бизнес Букс») and 37% speech-like. So the marks
# are dropped only around short spans, and kept where they carry a real pause.
QUOTE_PAIRS = [("«", "»"), ("\u201c", "\u201d"), ("\u201e", "\u201c"), ('"', '"')]
TITLE_MAX_WORDS = 5
TITLE_MAX_CHARS = 48


def strip_title_quotes(text: str) -> tuple[str, int]:
    """Remove quotation marks around short name/title spans.

    Leaves longer spans and anything containing sentence punctuation alone --
    those are quoted speech, where the pause belongs.
    """
    n = 0
    for open_q, close_q in QUOTE_PAIRS:
        if open_q == close_q:
            pattern = re.compile(
                rf"{re.escape(open_q)}([^{re.escape(open_q)}]{{1,{TITLE_MAX_CHARS}}}){re.escape(close_q)}")
        else:
            pattern = re.compile(
                rf"{re.escape(open_q)}([^{re.escape(open_q)}{re.escape(close_q)}]"
                rf"{{1,{TITLE_MAX_CHARS}}}){re.escape(close_q)}")

        def sub(m: re.Match) -> str:
            nonlocal n
            inner = m.group(1)
            if re.search(r"[.!?…;]", inner):
                return m.group(0)
            if len(inner.split()) > TITLE_MAX_WORDS:
                return m.group(0)
            n += 1
            return inner

        text = pattern.sub(sub, text)
    return text, n


# --------------------------------------------------------------------------
# number / designator expansion
# --------------------------------------------------------------------------

def _inflect(words: str, case: str, lang: str) -> str:
    """Hook for morphological agreement. Currently identity.

    Russian numerals must agree in case with their governing preposition
    ("в 1991 году" -> "в тысяча девятьсот девяносто первом году"). Doing this
    properly needs pymorphy3. Measured frequency in this library is low enough
    that nominative output is an acceptable Phase 1 compromise; revisit if a
    non-fiction book with dense dates enters the library.
    """
    return words


def _cardinal(n: int | float, lang: str) -> str:
    try:
        return num2words(n, lang="ru" if lang == "ru" else "en")
    except (NotImplementedError, OverflowError, ValueError):
        return str(n)


def _plural_ru(n: int, forms: tuple[str, str, str]) -> str:
    """Russian numeric agreement: 1 год / 2 года / 5 лет."""
    n = abs(n) % 100
    if 11 <= n <= 14:
        return forms[2]
    n %= 10
    if n == 1:
        return forms[0]
    if 2 <= n <= 4:
        return forms[1]
    return forms[2]


def _spell_letters(s: str, lang: str) -> str:
    table = RU_LETTER_NAMES if lang == "ru" else EN_LETTER_NAMES
    return " ".join(table.get(ch.lower(), ch) for ch in s)


def expand_numbers(text: str, lang: str) -> tuple[str, list[str]]:
    """Expand digits to words. Returns (text, notes) where notes flag guesses."""
    notes: list[str] = []
    letters = RU_LETTERS if lang == "ru" else "a-z"
    lcls = f"[{letters}]" if lang == "ru" else "[a-z]"

    # 1. Designators: 20Б-40, 107ИН, 64А, 32В, MP-35, ASD324DDD4E3C1.
    #    Short ones read naturally as a number plus letters ("compartment
    #    sixty-four A"). Long or repeatedly-alternating ones are serial
    #    numbers, where digit-by-digit is the only correct reading -- a human
    #    says "three two four", never "three hundred and twenty-four".
    def designator(m: re.Match) -> str:
        buf = m.group(0)
        runs = re.findall(rf"\d+|{lcls}+", buf, re.IGNORECASE)
        transitions = max(len(runs) - 1, 0)
        alnum = len(re.sub(r"[^0-9A-Za-z\u0400-\u04FF]", "", buf))
        # Deliberately conservative. "20Б-40" (a planet) reads better as
        # "twenty B forty" than as digit-by-digit, so a serial needs to be
        # clearly longer or more fragmented than a plain designator.
        serial = transitions >= 3 or alnum > 6

        out = []
        for part in re.split(r"([-/])", buf):
            if part in "-/":
                if lang == "ru":
                    out.append("тире" if part == "-" else "дробь")
                else:
                    out.append("dash" if part == "-" else "slash")
                continue
            for tok in re.findall(rf"\d+|{lcls}+", part, re.IGNORECASE):
                if not tok.isdigit():
                    out.append(_spell_letters(tok, lang))
                elif serial:
                    out.append(" ".join(_cardinal(int(d), lang) for d in tok))
                else:
                    out.append(_cardinal(int(tok), lang))
        return " ".join(out)

    # Matches runs that mix digits and letters in either order, including
    # alternating forms like 4E3C1 that a digits-then-letters pattern misses.
    pat_desig = re.compile(
        rf"\b(?:\d+{lcls}+|{lcls}+\d+)(?:[-/]?(?:\d+|{lcls}+))*\b", re.IGNORECASE)
    text, n = pat_desig.subn(designator, text)
    if n:
        notes.append(f"{n} alphanumeric designator(s) spelled out")

    # 2. Fractions written with a slash: 3/4 -> три четверти (approximate).
    def frac(m: re.Match) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        if lang == "ru":
            names = {2: "вторых", 3: "третьих", 4: "четвертых", 5: "пятых",
                     8: "восьмых", 10: "десятых"}
            if a == 3 and b == 4:
                return "три четверти"
            if a == 1 and b == 2:
                return "одна вторая"
            return f"{_cardinal(a, lang)} {names.get(b, _cardinal(b, lang))}"
        return f"{_cardinal(a, lang)} over {_cardinal(b, lang)}"

    text = re.sub(r"\b(\d{1,3})/(\d{1,3})\b", frac, text)

    # 3. Decimals -- the corpus mixes ',' and '.' as the separator. num2words
    #    handles Russian place names and gender agreement ("три целых две
    #    десятых"), so delegate rather than reimplement.
    def dec(m: re.Match) -> str:
        whole, frac_part = m.group(1), m.group(3)
        try:
            return num2words(float(f"{whole}.{frac_part}"),
                             lang="ru" if lang == "ru" else "en")
        except (NotImplementedError, OverflowError, ValueError):
            return m.group(0)

    text = re.sub(r"\b(\d+)([.,])(\d+)\b", dec, text)

    # 4. Plain integers. The governing noun is left untouched: in running
    #    prose it already agrees with the numeral the author wrote, and
    #    rewriting it correctly would need morphology (see _inflect).
    text = re.sub(r"\b\d+\b", lambda m: _cardinal(int(m.group(0)), lang), text)

    leftover = re.findall(r"\d", text)
    if leftover:
        notes.append(f"{len(leftover)} digit(s) survived expansion")
    return text, notes


def expand_abbreviations(text: str, lang: str) -> str:
    table = RU_ABBREV if lang == "ru" else EN_ABBREV
    # The leading guard is essential: a bare re.escape("в.") also matches the
    # tail of "градусов.", yielding "градусовек".
    guard = r"(?<![^\W\d_])" if lang == "ru" else r"(?<![A-Za-z])"
    for k in sorted(table, key=len, reverse=True):
        text = re.sub(guard + re.escape(k), table[k], text,
                      flags=re.IGNORECASE if lang == "ru" else 0)
    return text


# --------------------------------------------------------------------------
# English-specific expansion
# --------------------------------------------------------------------------

# Measured on Old Man's War (505k chars): "CDF" occurs 147 times, plus PDA,
# DNA, MP, CDFS. Read as words these are gibberish, so acronyms are spelled
# out -- but "ONE"/"TWO"/"NEW"/"YOU" also appear in caps as chapter titles and
# emphasis, and spelling those out would be worse. The test is therefore
# "is this a real word?", not "is this uppercase?".

SYSTEM_WORDLIST = "/usr/share/dict/words"

# Portable fallback: short words that plausibly appear fully capitalised in
# prose. Used when no system word list exists (e.g. a bare Linux container).
_FALLBACK_WORDS = set("""
a i an as at be by do go he if in is it me my no of on or so to up us we
all and any are but can day did end for get got had has her him his how
its let man may men new not now off old one our out own put run saw say
see she six ten the too two use war was way who why yes yet you
"""
.split()) | set("""
about above after again ahead alone along also away back been before began
being best both call came come dark days dead does done down each even ever
face fact fall feel felt find fire first five four from full gave give gone
good half hand hard have head hear held help here high hold home hope hour
house into just keep kept kind knew know last late left less life like line
little live long look lost love made make many mean meet mind more most move
much must name near need never next nice night none only open over part pass
past play point read real rest room said same seem seen sent side since
small some soon sort stay step still stop such sure take talk tell than that
them then there these they thing think this those three time told took turn
under upon very wait walk want well went were what when where which while
white will wish with word work would year your
""".split())

# Real dictionary words that are nonetheless acronyms when capitalised.
_ACRONYM_OVERRIDES = {"ID", "US", "UK", "UN", "AI", "TV", "PM", "AM"}
_SPOKEN_AS_WORD = {"OK": "okay", "OKAY": "okay", "NASA": "NASA", "SETI": "SETI"}

# Roman numerals are only recognised from this closed set. The tempting general
# solution is wrong: CD, MD, DC, MC, MI, DI, LI and XL are all valid roman
# numerals *and* plausible acronyms, and "ID" parses as 499. Restricting to
# forms that realistically appear as sequence numbers avoids inventing numbers
# out of initialisms.
_ROMAN_SAFE = {
    "II": 2, "III": 3, "IV": 4, "VI": 6, "VII": 7, "VIII": 8, "IX": 9,
    "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16,
    "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20, "XXI": 21,
}

_EN_LETTER_SAY = {
    "a": "ay", "b": "bee", "c": "see", "d": "dee", "e": "ee", "f": "eff",
    "g": "gee", "h": "aitch", "i": "eye", "j": "jay", "k": "kay", "l": "el",
    "m": "em", "n": "en", "o": "oh", "p": "pee", "q": "cue", "r": "ar",
    "s": "ess", "t": "tee", "u": "you", "v": "vee", "w": "double-you",
    "x": "ex", "y": "why", "z": "zee",
}

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

_wordset: set[str] | None = None


def _english_words() -> set[str]:
    """Load the system word list once, falling back to the built-in set.

    The fallback matters for portability: macOS ships /usr/share/dict/words but
    a minimal Linux image (the Spark's container) generally does not.
    """
    global _wordset
    if _wordset is None:
        words = set(_FALLBACK_WORDS)
        try:
            with open(SYSTEM_WORDLIST, encoding="utf-8", errors="ignore") as fh:
                words |= {w.strip().lower() for w in fh if 1 < len(w.strip()) <= 6}
        except OSError:
            pass
        _wordset = words
    return _wordset


def _roman_to_int(s: str) -> int | None:
    total = prev = 0
    for ch in reversed(s.upper()):
        v = _ROMAN.get(ch)
        if v is None:
            return None
        total += -v if v < prev else v
        prev = max(prev, v)
    return total or None


def clock_words(h: int, mn: int, meridiem: str = "") -> str:
    """Spoken form of a clock time, shared with the verifier so both sides of
    the comparison produce the same words."""
    mer = meridiem.replace(".", "").replace(" ", "").lower()
    out = num2words(h, lang="en")
    if mn == 0:
        out += " o'clock"
    elif mn < 10:
        out += " oh " + num2words(mn, lang="en")
    else:
        out += " " + num2words(mn, lang="en")
    if mer.startswith("a"):
        out += " in the morning"
    elif mer.startswith("p"):
        out += " in the evening" if h >= 6 else " in the afternoon"
    return out


def expand_english(text: str) -> tuple[str, list[str]]:
    """Acronyms, military time and roman numerals -- in that order."""
    notes: list[str] = []
    words = _english_words()

    # 0. Times of day: "8:30 A.M." Without this the colon survives into the
    #    text and the model reads "eight:thirty".
    def clock(m: re.Match) -> str:
        return clock_words(int(m.group(1)), int(m.group(2)), m.group(3) or "")

    text, n_clock = re.subn(
        r"\b(\d{1,2}):([0-5]\d)(?:\s*([AaPp]\.?[Mm]\.?))?", clock, text)
    if n_clock:
        notes.append(f"{n_clock} time(s) of day expanded")

    # 1. Military time. A leading zero is unambiguous ("0600"); a bare "1200"
    #    is only a time when the text says so, otherwise it is a quantity.
    def miltime(m: re.Match) -> str:
        h, mn = int(m.group(1)[:2]), int(m.group(1)[2:])
        hh = "oh " + num2words(h, lang="en") if h < 10 else num2words(h, lang="en")
        return f"{hh} hundred" if mn == 0 else f"{hh} {num2words(mn, lang='en')}"

    text, n = re.subn(r"\b(0[0-9][0-5][0-9])\b(?!\s*(?:kilometers|meters|feet))",
                      miltime, text)
    if n:
        notes.append(f"{n} military time(s) expanded")
    text, n2 = re.subn(r"\b([01][0-9][0-5][0-9])\b(?=\s+hours?\b)", miltime, text)
    n += n2

    # 2. Acronyms: 2-5 uppercase letters that are not English words.
    changed = 0

    def one_token(m: re.Match) -> str:
        nonlocal changed
        whole = m.group(0)
        tok = re.match(r"[A-Z]+", whole).group(0)
        plural = whole.endswith("s")

        if tok in _SPOKEN_AS_WORD:
            changed += 1
            return _SPOKEN_AS_WORD[tok] + ("s" if plural else "")
        # Roman numerals first, but only the unambiguous ones.
        if tok in _ROMAN_SAFE and not plural:
            changed += 1
            return num2words(_ROMAN_SAFE[tok], lang="en")
        # A real English word in caps is emphasis or a title, not an acronym.
        if tok not in _ACRONYM_OVERRIDES and tok.lower() in words:
            return whole
        changed += 1
        said = "-".join(_EN_LETTER_SAY.get(c.lower(), c) for c in tok)
        if plural:
            # "eye-dee" + "es" reads as "eye-deees"; a trailing e takes bare s.
            said += "s" if said.endswith(("e", "y")) else "es"
        return said

    text = re.sub(r"\b[A-Z]{2,5}s?\b", one_token, text)
    if changed:
        notes.append(f"{changed} acronym/roman token(s) expanded")
    return text, notes


# --------------------------------------------------------------------------
# sentence segmentation
# --------------------------------------------------------------------------

_ABBR_GUARD = re.compile(r"\b([A-Za-zА-Яа-яЁё]{1,5})\.\s*$")


def split_sentences(text: str, lang: str) -> list[str]:
    """Sentence split tuned for prose, including Russian dialogue dashes.

    Guards against splitting on initials ("А. С. Пушкин"), known abbreviations,
    and ellipses, all of which are common in this corpus.
    """
    non_terminal = RU_NON_TERMINAL if lang == "ru" else EN_NON_TERMINAL
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Protect ellipses and decimal-looking leftovers from the splitter.
    text = text.replace("...", "…")

    out: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ".!?…":
            # Consume runs like "?!" or "!.."
            j = i
            while j + 1 < n and text[j + 1] in ".!?…":
                j += 1
            nxt = text[j + 1:j + 3]
            candidate = text[start:j + 1]

            # Closing quote/bracket belongs to this sentence.
            closed_quote = False
            k = j + 1
            while k < n and text[k] in '"»”’\')':
                closed_quote = True
                k += 1
                candidate = text[start:k]
                j = k - 1
                nxt = text[j + 1:j + 3]

            m = _ABBR_GUARD.search(candidate)
            is_abbrev = bool(m) and m.group(1).lower() in non_terminal
            is_initial = bool(re.search(r"\b[A-ZА-ЯЁ]\.\s*$", candidate))
            # A sentence must be followed by space + capital / dash / quote.
            follows = re.match(r"\s+[A-ZА-ЯЁ«\"—–\-—]", nxt) or j + 1 >= n
            # ...but a dash straight after a closing quote is Russian dialogue
            # attribution («Стой!» — крикнул он.), which is one sentence.
            if closed_quote and re.match(r"\s*[—–-]", nxt):
                follows = False

            if not is_abbrev and not is_initial and follows:
                s = candidate.strip()
                if s:
                    out.append(s)
                start = j + 1
            i = j + 1
            continue
        i += 1

    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return [s.replace("…", "...") for s in out]


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    index: int
    text: str
    chapter_index: int
    # Silence to append after this chunk, ms. Encodes prose structure as pacing.
    pause_ms: int
    is_paragraph_end: bool

    @property
    def char_count(self) -> int:
        return len(self.text)


PAUSE_SENTENCE = 350
PAUSE_PARAGRAPH = 700


def _hard_split(sentence: str, limit: int) -> list[str]:
    """Last-resort split of an over-long sentence, at the best available seam."""
    if len(sentence) <= limit:
        return [sentence]
    parts: list[str] = []
    rest = sentence
    # Prefer clause seams, in descending quality.
    seams = [" — ", "; ", ", ", " "]
    while len(rest) > limit:
        cut = -1
        for seam in seams:
            cut = rest.rfind(seam, MIN_CHUNK_CHARS, limit)
            if cut > 0:
                cut += len(seam)
                break
        if cut <= 0:
            cut = limit
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return parts


def chunk_paragraphs(
    paragraphs: list[str], lang: str, chapter_index: int,
    max_chars: int = MAX_CHUNK_CHARS, start_index: int = 0,
) -> tuple[list[Chunk], list[str]]:
    """Pack sentences into chunks, never crossing a sentence boundary."""
    notes: list[str] = []
    chunks: list[Chunk] = []
    idx = start_index

    for p_i, para in enumerate(paragraphs):
        para = expand_abbreviations(para, lang)
        n_notes = []
        para, n_pron = apply_pronunciations(para)
        if n_pron:
            n_notes.append(f"{n_pron} pronunciation respelling(s)")
        para, n_quotes = strip_title_quotes(para)
        if n_quotes:
            n_notes.append(f"{n_quotes} title/name quote(s) unquoted for pacing")
        if lang == "en":
            para, en_notes = expand_english(para)
            n_notes += en_notes
        para, num_notes = expand_numbers(para, lang)
        n_notes += num_notes
        notes.extend(f"ch{chapter_index} p{p_i}: {x}" for x in n_notes)
        sentences = split_sentences(para, lang)
        if not sentences:
            continue

        buf: list[str] = []
        buf_len = 0
        for s_i, sent in enumerate(sentences):
            pieces = _hard_split(sent, max_chars)
            if len(pieces) > 1:
                notes.append(
                    f"ch{chapter_index} p{p_i}: sentence of {len(sent)} chars "
                    f"hard-split into {len(pieces)}")
            for piece in pieces:
                if buf and buf_len + 1 + len(piece) > max_chars:
                    idx += 1
                    chunks.append(Chunk(idx, " ".join(buf), chapter_index,
                                        PAUSE_SENTENCE, False))
                    buf, buf_len = [], 0
                buf.append(piece)
                buf_len += len(piece) + 1

        if buf:
            idx += 1
            chunks.append(Chunk(idx, " ".join(buf), chapter_index,
                                PAUSE_PARAGRAPH, True))
        elif chunks:
            chunks[-1].pause_ms = PAUSE_PARAGRAPH
            chunks[-1].is_paragraph_end = True

    return chunks, notes

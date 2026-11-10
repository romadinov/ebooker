"""Latin script inside a Russian book: who reads it, and how.

Russian prose quotes brand names («Coca-Cola»), drops in English words for
effect, and occasionally switches to English outright. The Russian speech
models read none of it -- Shtorm-PocketTTS-RU skips Latin words altogether,
which left 99 flagged chunks in one chapter of Б. Вавилонская, a satire that
drifts from «Marlboro» through pidgin («Парламент достиг consensus'a, и не
long, совсем не long оказался way до Типерери!») into pages of broken English.

Two treatments, chosen per sentence by how much of it is Latin:

* A sentence that is mostly Latin is English and is routed to an English voice
  (``Chunk.lang == "en"``). Pocket TTS has an English model that clones from
  the same reference clip, so the narrator keeps his voice.
* A Russian sentence with Latin words in it stays with the Russian voice, and
  the Latin words are transliterated into Cyrillic the way a Russian narrator
  says them: «Кока-Кола», «Мальборо», «консенсуса», «не лонг, совсем не лонг».
  That is also how the author meant the pidgin to sound.

Transliteration goes through the CMU pronouncing dictionary (ARPAbet) and the
usual Russian practical transcription of English sounds, so it is driven by
pronunciation, not spelling: ``tough`` -> «таф», ``Cherokee`` -> «Чероки».
The dictionary's primary stress becomes a combining acute on the Cyrillic
vowel, which the Russian stress marker keeps (hand marks win over its own
guess), so «Кэ́мел» is not left for silero to guess at. Reduced vowels follow
the spelling, as Russian practice does («резиденс», «докюментс»). Words the dictionary
lacks are tried without a plural/past/-ing suffix, then transliterated by
spelling. ``pronunciation.txt`` overrides anything heard wrong, as always:
it is applied before this runs.
"""

from __future__ import annotations

import re
from functools import lru_cache

ACUTE = "́"

LATIN = re.compile(r"[A-Za-z]")
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
# A word with inner apostrophes: it's, McDonald's, consensus'a (the author's
# Russian ending in Latin letters).
LATIN_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
CYR_VOWELS = "аеёиоуыэюя"

# Share of Latin among a sentence's letters at which it counts as English and
# goes to the English voice. High on purpose: a pidgin sentence («Простым
# people to жрать was nothing!») is the Russian narrator's to read, with his
# accent, as the author meant it. The few Cyrillic words left in a sentence
# that does qualify are romanised for the English model, which reads no Cyrillic.
ENGLISH_SHARE = 0.9

# A Roman numeral in Russian prose is an ordinal («XX век», «Пётр I»), not an
# English word; its case follows the word after it.
ROMAN = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
CASE_BY_NOUN = {"века": "gen", "в.": "gen", "веке": "prep", "веку": "dat", "веком": "ins"}
ORDINAL_ENDINGS = {  # nominative masculine ending -> the other cases
    "ый": {"gen": "ого", "prep": "ом", "dat": "ому", "ins": "ым"},
    "ой": {"gen": "ого", "prep": "ом", "dat": "ому", "ins": "ым"},
    "ий": {"gen": "ьего", "prep": "ьем", "dat": "ьему", "ins": "ьим"},
}

ROMANIZE = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
}

# English letter names: a short all-caps token (VIP, TV, BMW) is read letter
# by letter, the way Russians say them.
LETTER_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи",
    "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн",
    "o": "оу", "p": "пи", "q": "кью", "r": "ар", "s": "эс", "t": "ти", "u": "ю",
    "v": "ви", "w": "дабл-ю", "x": "экс", "y": "уай", "z": "зед",
}

# ARPAbet -> Cyrillic. Vowels carry a second form for after a consonant where
# Russian practice softens: metal -> «метал», not «мэтал»; Lake -> «Лейк».
VOWELS = {
    "AA": ("а", "а"), "AE": ("э", "э"), "AH": ("а", "а"), "AO": ("о", "о"),
    "AW": ("ау", "ау"), "AY": ("ай", "ай"), "EH": ("э", "е"), "ER": ("эр", "ер"),
    "EY": ("эй", "ей"), "IH": ("и", "и"), "IY": ("и", "и"), "OW": ("оу", "оу"),
    "OY": ("ой", "ой"), "UH": ("у", "у"), "UW": ("у", "у"),
    # OW is «о» in Russian practice (Кока-Кола, казино); «оу» only when it
    # ends a stressed word (шоу). Handled in _from_phones.
}
# Y + vowel is one iotated letter.
IOTATED = {"AA": "я", "AE": "я", "AH": "я", "AO": "йо", "EH": "е", "ER": "ер",
           "EY": "ей", "IH": "йи", "IY": "йи", "OW": "йоу", "UH": "ю", "UW": "ю",
           "AW": "яу", "AY": "яй", "OY": "йой"}
CONSONANTS = {
    "B": "б", "CH": "ч", "D": "д", "DH": "з", "F": "ф", "G": "г", "HH": "х",
    "JH": "дж", "K": "к", "L": "л", "M": "м", "N": "н", "NG": "нг", "P": "п",
    "R": "р", "S": "с", "SH": "ш", "T": "т", "TH": "т", "V": "в", "W": "в",
    "Y": "й", "Z": "з", "ZH": "ж",
}
VOICELESS = {"P", "T", "K", "F", "S", "SH", "CH", "TH"}

# Spelling fallback for words outside the dictionary. Digraphs first.
GRAPHEMES = [
    ("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"), ("ck", "к"), ("qu", "кв"),
    ("wh", "в"), ("ng", "нг"), ("oo", "у"), ("ee", "и"), ("ea", "и"), ("ou", "ау"),
    ("ow", "оу"), ("ay", "эй"), ("ey", "ей"), ("ai", "эй"), ("oi", "ой"), ("oy", "ой"),
    ("a", "а"), ("b", "б"), ("c", "к"), ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"),
    ("h", "х"), ("i", "и"), ("j", "дж"), ("k", "к"), ("l", "л"), ("m", "м"), ("n", "н"),
    ("o", "о"), ("p", "п"), ("q", "к"), ("r", "р"), ("s", "с"), ("t", "т"), ("u", "у"),
    ("v", "в"), ("w", "в"), ("x", "кс"), ("y", "и"), ("z", "з"),
]


def latin_share(text: str) -> float:
    """Latin letters as a share of all Latin+Cyrillic letters; 0 for none."""
    lat = len(LATIN.findall(text))
    cyr = len(CYRILLIC.findall(text))
    return lat / (lat + cyr) if lat + cyr else 0.0


def is_english(sentence: str) -> bool:
    return latin_share(sentence) >= ENGLISH_SHARE


@lru_cache(maxsize=1)
def _cmu() -> dict[str, list[list[str]]]:
    import cmudict
    return cmudict.dict()


def _lookup(word: str) -> list[str] | None:
    """ARPAbet phones for a lower-case word, trying common suffixes off."""
    d = _cmu()
    if word in d:
        return list(d[word][0])
    if len(word) > 3:
        for suffix, tail in (
            ("'s", ["Z"]), ("’s", ["Z"]), ("s'", ["Z"]), ("es", ["IH0", "Z"]),
            ("s", ["Z"]), ("ed", ["D"]), ("ing", ["IH0", "NG"]),
        ):
            stem = word[: -len(suffix)] if word.endswith(suffix) else None
            if stem and stem in d:
                phones = list(d[stem][0])
                if tail == ["Z"] and phones and phones[-1] in VOICELESS:
                    tail = ["S"]
                if tail == ["D"] and phones and phones[-1] in VOICELESS:
                    tail = ["T"]
                if tail == ["D"] and phones and phones[-1] in ("T", "D"):
                    tail = ["IH0", "D"]
                return phones + tail
    return None


# Reduced vowels (the schwa and its neighbours) are spelt by letter in Russian
# practice, not by sound: residence -> «резиденс», not «резиданс»; documents
# -> «докюментс». The dictionary gives the sounds, the spelling the letters,
# and the two line up whenever the word has as many vowel-letter groups as
# vowel sounds (a silent final e allowed). Otherwise the sounds alone decide.
REDUCED = {"AH", "IH", "ER"}
BY_LETTER = {"a": "а", "e": "е", "i": "и", "o": "о", "u": "у", "y": "и"}
IOTATED_BY_LETTER = {"a": "я", "e": "е", "i": "йи", "o": "йо", "u": "ю", "y": "йи"}
VOWEL_GROUP = re.compile(r"[aeiouy]+")


def _vowel_letters(spelling: str, n_vowels: int) -> list[str] | None:
    groups = VOWEL_GROUP.findall(spelling.lower())
    if len(groups) == n_vowels + 1 and spelling.lower().endswith("e") and len(groups) > 1:
        groups = groups[:-1]
    return groups if len(groups) == n_vowels else None


def _from_phones(phones: list[str], spelling: str = "") -> str:
    def is_vowel(p: str) -> bool:
        return p[:-1] in VOWELS if p[-1].isdigit() else p in VOWELS

    letters = _vowel_letters(spelling, sum(is_vowel(p) for p in phones)) if spelling else None
    out: list[str] = []
    stressed_at: int | None = None
    after_consonant = False
    v = 0        # vowel sounds seen so far, indexes ``letters``
    i = 0
    while i < len(phones):
        p = phones[i]
        base, stress = (p[:-1], p[-1]) if p[-1].isdigit() else (p, "")
        iotate = base == "Y" and i + 1 < len(phones) and is_vowel(phones[i + 1])
        if iotate:
            i += 1
            p = phones[i]
            base, stress = p[:-1], p[-1]
        if base in VOWELS:
            letter = letters[v][0] if letters else ""
            v += 1
            if iotate:
                cyr = IOTATED_BY_LETTER.get(letter, IOTATED[base]) if stress == "0" else IOTATED[base]
            elif stress == "0" and base in REDUCED and letter:
                cyr = BY_LETTER[letter] + ("р" if base == "ER" else "")
            elif base in ("AA", "AO") and letter == "o":
                cyr = "о"
            elif base == "OW":
                cyr = "оу" if (stress == "1" and i == len(phones) - 1) else "о"
            else:
                cyr = VOWELS[base][1 if after_consonant else 0]
            out.append(cyr)
            if stress == "1" and stressed_at is None:
                stressed_at = len(out) - 1
            after_consonant = False
        elif base == "NG" and i + 1 < len(phones) and phones[i + 1] in ("K", "G"):
            out.append("н")
            after_consonant = True
        else:
            out.append(CONSONANTS.get(base, ""))
            after_consonant = True
        i += 1
    # The acute follows the vowel letter; for a diphthong, its first element.
    if stressed_at is not None and sum(ch in CYR_VOWELS for ch in "".join(out)) >= 2:
        vow = out[stressed_at]
        k = next((j + 1 for j, ch in enumerate(vow) if ch in CYR_VOWELS), len(vow))
        out[stressed_at] = vow[:k] + ACUTE + vow[k:]
    return "".join(out)


def _from_spelling(word: str) -> str:
    w = word.lower()
    # A silent final e: "Hilton" has none, "Plaza" ends in a, "Lake" drops it.
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiou" and any(c in "aeiou" for c in w[:-2]):
        w = w[:-1]
    out: list[str] = []
    i = 0
    while i < len(w):
        for g, cyr in GRAPHEMES:
            if w.startswith(g, i):
                if g == "c" and i + 1 < len(w) and w[i + 1] in "eiy":
                    cyr = "с"
                out.append(cyr)
                i += len(g)
                break
        else:
            i += 1
    return "".join(out)


def _match_case(src: str, cyr: str) -> str:
    if not cyr:
        return cyr
    if src.isupper() and len(src) > 1:
        return cyr.upper()
    if src[0].isupper():
        return cyr[0].upper() + cyr[1:]
    return cyr


def transliterate_word(word: str) -> str:
    """One Latin token (letters and inner apostrophes) in Cyrillic."""
    lower = word.lower()
    if word.isupper() and 2 <= len(word) <= 4:
        return "-".join(LETTER_NAMES[c] for c in lower)
    phones = _lookup(lower)
    if phones is not None:
        return _match_case(word, _from_phones(phones, word))
    if "'" in word or "’" in word:
        parts = re.split(r"['’]", word)
        return "".join(transliterate_word(p) if p else "" for p in parts)
    return _match_case(word, _from_spelling(word))


def roman_ordinal(numeral: str, case: str = "nom") -> str | None:
    """«XX» -> «двадцатый», «двадцатого» in the genitive; None if not a numeral."""
    if not numeral or not ROMAN.match(numeral):
        return None
    value = 0
    prev = 0
    for ch in reversed(numeral):
        v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}[ch]
        value += -v if v < prev else v
        prev = max(prev, v)
    from num2words import num2words
    word = num2words(value, lang="ru", to="ordinal")
    if case != "nom":
        for nom, forms in ORDINAL_ENDINGS.items():
            if word.endswith(nom):
                return word[: -len(nom)] + forms[case]
    return word


def transliterate(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace every Latin word in ``text``; returns the text and (from, to) pairs."""
    changes: list[tuple[str, str]] = []

    def sub(m: re.Match) -> str:
        src = m.group(0)
        if ROMAN.match(src):
            rest = text[m.end():]
            case_m = re.match(r"\s+(век[а-я]*|в\.)", rest)
            case = CASE_BY_NOUN.get(case_m.group(1), "nom") if case_m else "nom"
            out = roman_ordinal(src, case) or src
        else:
            out = transliterate_word(src)
        changes.append((src, out))
        return out

    return LATIN_WORD.sub(sub, text), changes


def romanize(text: str) -> str:
    """Cyrillic letters left in an English sentence, in Latin for the English voice."""
    out = []
    for ch in text:
        low = ch.lower()
        if low in ROMANIZE:
            lat = ROMANIZE[low]
            out.append(lat.capitalize() if ch.isupper() else lat)
        else:
            out.append(ch)
    return "".join(out)

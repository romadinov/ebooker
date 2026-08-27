"""Explicit Russian stress marking for Silero, via RUAccent.

Why this exists: Silero's built-in `put_accent` places stress itself, and it
gets words wrong. The ASR round-trip cannot catch that -- Whisper transcribes
за́мок and замо́к as the same string "замок" -- so a 0.0% CER says the words were
right, not the stress. Wrong stress is only detectable by listening, or by
comparing against a dedicated accentuation model, which is what this does.

Silero accepts explicit '+' before the stressed vowel (its own symbol set
includes '+'), so the fix is to mark the text ourselves with RUAccent and pass
put_accent=False so Silero does not overrule us.

Also carries a shim for a version skew in ruaccent 1.5.8.x: the accentuator
ONNX published on the Hub expects a token_type_ids input that the pip package
does not supply, which raises
    ValueError: Required inputs (['token_type_ids']) are missing
on any word that falls through to the neural accentuator.
"""

from __future__ import annotations

import re

STRESS = "+"
VOWELS = "аеёиоуыэюя"

# Unstressed clitics. Marking these produces a staccato, over-emphatic reading:
# Russian prepositions, conjunctions and particles lean on the following word
# rather than carrying their own stress. Content monosyllables (Дон, нос, знал)
# are a different matter and must keep their mark.
CLITICS = {
    "а", "и", "но", "да", "же", "ли", "бы", "б", "не", "ни", "то",
    "на", "за", "по", "до", "от", "из", "у", "о", "об", "во", "со", "ко",
    "как", "что", "чем", "уж", "ведь", "вот", "хоть", "лишь", "аж",
    "их", "его", "ее", "её", "мне", "мной", "нас", "вас", "им", "ей", "ему",
}


def _patch_accent_model() -> None:
    """Supply token_type_ids when the ONNX signature demands it."""
    import numpy as np
    from ruaccent import accent_model as am

    if getattr(am.AccentModel, "_ebooker_patched", False):
        return
    original = am.AccentModel.put_accent

    def put_accent(self, word):
        try:
            return original(self, word)
        except ValueError as e:
            if "token_type_ids" not in str(e):
                raise
            lower = word.lower()
            inputs = self.tokenizer(lower, return_tensors="np")
            inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
            needed = {i.name for i in self.session.get_inputs()}
            if "token_type_ids" in needed and "token_type_ids" not in inputs:
                inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
            inputs = {k: v for k, v in inputs.items() if k in needed}
            outputs = self.session.run(None, inputs)
            names = {o.name: i for i, o in enumerate(self.session.get_outputs())}
            logits = outputs[names["logits"]]
            probs = am.softmax(logits)
            scores = np.max(probs, axis=-1)[0]
            labels = np.argmax(logits, axis=-1)[0]
            pred = [{"label": self.id2label[str(l)], "score": float(s)}
                    for l, s in zip(labels, scores)]
            return self.render_stress(word, pred)

    am.AccentModel.put_accent = put_accent
    am.AccentModel._ebooker_patched = True


# Words RUAccent gets wrong regardless of case. Its 3.19M-entry dictionary
# maps "после" to "посл+е", which is simply incorrect -- the stress is always
# on the first syllable. Entries here win over the model.
OVERRIDES: dict[str, str] = {
    "после": "п+осле",
}

# Phrase-level corrections, for words whose stress is genuinely ambiguous and
# therefore cannot be fixed by a single-word override. "ноги" is ноги́ in the
# genitive singular but но́ги in the nominative/accusative plural; RUAccent reads
# the idiom "на ноги" (accusative, "to one's feet") as a genitive and stresses
# it "ног+и". Applied after marking, so these win over the model.
#
# Each entry is (pattern, replacement) on already-marked text; \1 etc. work.
PREPOSITIONS = r"(?:в|во|из|изо|на|над|под|у|к|ко|до|от|ото|за|через|возле|около|внутри|сквозь)"

PHRASE_OVERRIDES: list[tuple[str, str]] = [
    # на/в ноги -- accusative plural: но́ги, not the genitive singular ноги́
    (r"\b(на|в)\s+ног\+и\b", r"\1 н+оги"),
    (r"\bног\+и\b(?=\s*(?:как|и|,|\.|$))", "н+оги"),

    # отсе́к (a compartment) versus отсёк (past tense of отсечь). RUAccent reads
    # the bare nominative as the verb, which is right for "Он отсек ему руку"
    # but wrong for "отсек 107" -- and this book is full of ship compartments.
    # Disambiguated by context rather than by a blanket override:
    #   after a preposition, or followed by a number, it is the noun.
    (rf"\b({PREPOSITIONS})(\s+)отс\+ёк\b", r"\1\2отс+ек"),
    (r"\bотс\+ёк\b(?=\s+(?:\d|[+А-Яа-яЁё]*(?:надцать|дцать|десят|сто|ноль)))",
     "отс+ек"),
    (r"\b(\w*ый|\w*ой|\w*ий|этот|тот|весь|каждый|данный)(\s+)отс\+ёк\b",
     r"\1\2отс+ек"),
]

_SENT_START = re.compile(r"(?:^|(?<=[.!?…])\s+|(?<=\n))\s*([А-ЯЁ][а-яё]+)")


class Accentuator:
    """RUAccent wrapper producing Silero-compatible '+' marked text.

    Two corrections sit on top of the library:

    1. **Sentence-initial words are accentuated in lower case.** RUAccent's
       dictionary is case-sensitive, so "После" misses it entirely and falls
       through to the neural accentuator, which mis-stresses it ("Посл+е") or
       returns no mark at all ("Однако"). Only sentence-initial words are
       lowered -- mid-sentence capitals are proper nouns, where case can be
       lexically meaningful, so those are left alone.
    2. **An override dictionary** for words the library gets wrong outright.
    """

    def __init__(self, omograph_model_size: str = "turbo3.1",
                 use_dictionary: bool = True, tiny_mode: bool = False,
                 overrides: dict[str, str] | None = None,
                 overrides_file: str | None = None):
        _patch_accent_model()
        from ruaccent import RUAccent

        self.overrides = {**OVERRIDES, **load_overrides_file(overrides_file),
                          **(overrides or {})}
        self.model = RUAccent()
        self.model.load(omograph_model_size=omograph_model_size,
                        use_dictionary=use_dictionary, tiny_mode=tiny_mode,
                        custom_dict=dict(self.overrides))

    def mark(self, text: str) -> str:
        """Two passes, spliced.

        Lowering the sentence-initial word also changes the sentence context the
        homograph model sees, which measurably degraded *other* words in the
        same sentence. So the original text supplies every word's stress, and
        the lowered pass is consulted only for the sentence-initial position it
        was meant to fix.
        """
        primary = self.model.process_all(text)
        lowered, _ = _lower_sentence_starts(text)
        if lowered != text:
            secondary = self.model.process_all(lowered)
            primary = _splice_sentence_starts(primary, secondary)
        primary = _apply_overrides(primary, self.overrides)
        primary = _apply_phrase_overrides(primary)
        primary = _fill_gaps(primary, self.model)
        return _tidy(primary)


DEFAULT_OVERRIDES_FILE = "stress_overrides.txt"


def load_overrides_file(path: str | None = None) -> dict[str, str]:
    """Read user corrections from a plain text file.

    One per line, `word = m+arked`, blank lines and # comments ignored. This
    exists so corrections found by ear during a long library run do not require
    touching code -- expect to add entries as you listen.
    """
    import pathlib as _pl

    p = _pl.Path(path or DEFAULT_OVERRIDES_FILE)
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        bare, _, marked = line.partition("=")
        bare, marked = bare.strip().lower(), marked.strip()
        if bare and marked:
            out[bare] = marked
    return out


def _lower_sentence_starts(text: str) -> tuple[str, list[str]]:
    """Lower-case the first word of each sentence, remembering the originals."""
    restore: list[str] = []

    def sub(m: re.Match) -> str:
        word = m.group(1)
        restore.append(word)
        return m.group(0)[: m.start(1) - m.start()] + word.lower()

    return _SENT_START.sub(sub, text), restore


_WORD = re.compile(r"[+А-Яа-яЁё]+")


def _splice_sentence_starts(primary: str, secondary: str) -> str:
    """Take sentence-initial words from `secondary`, everything else from `primary`."""
    ptoks = _WORD.findall(primary)
    stoks = _WORD.findall(secondary)
    if len(ptoks) != len(stoks):
        return primary          # alignment lost; keep the original decisions

    # Index of the first word of each sentence, over the token sequence.
    starts: set[int] = set()
    expect = True
    for i, tok in enumerate(ptoks):
        if expect:
            starts.add(i)
            expect = False
        # Look at what follows this token in the source to spot a boundary.
        pos = primary.find(tok)
        if pos >= 0 and re.match(r"\s*[.!?…]", primary[pos + len(tok):]):
            expect = True
    if not starts:
        return primary

    out, idx = [], 0
    def sub(m: re.Match) -> str:
        nonlocal idx
        i = idx
        idx += 1
        if i not in starts:
            return m.group(0)
        repl = stoks[i]
        original = m.group(0)
        bare_o = original.replace(STRESS, "")
        bare_r = repl.replace(STRESS, "")
        if bare_o.lower() != bare_r.lower():
            return original
        # Carry the replacement's stress onto the original's capitalisation.
        marked_at = repl.find(STRESS)
        if marked_at == -1:
            return original
        letters = list(bare_o)
        vowel_idx = len(bare_r[:marked_at].replace(STRESS, ""))
        if vowel_idx >= len(letters):
            return original
        letters.insert(vowel_idx, STRESS)
        return "".join(letters)

    return _WORD.sub(sub, primary)


def _fill_gaps(marked: str, model) -> str:
    """Ensure every multi-vowel word carries a mark.

    With put_accent=False, an unmarked multi-vowel word gives Silero no stress
    information at all, which is how "после" became "после'" in the first place.
    Anything the dictionary and homograph model left bare is sent to the neural
    accentuator as a last resort.
    """
    def sub(m: re.Match) -> str:
        tok = m.group(0)
        if STRESS in tok:
            return tok
        bare = tok.replace(STRESS, "")
        nv = sum(c.lower() in VOWELS for c in bare)
        if nv < 1:
            return tok
        if "ё" in bare.lower():          # ё is inherently stressed
            return tok
        if bare.lower() in CLITICS:      # leans on the next word, no stress
            return tok
        try:
            guess = model.accent_model.put_accent(bare.lower())
        except Exception:
            return tok
        if STRESS not in guess:
            return tok
        at = guess.find(STRESS)
        letters = list(bare)
        if at >= len(letters):
            return tok
        letters.insert(at, STRESS)
        return "".join(letters)

    return _WORD.sub(sub, marked)


def _restore_case(marked: str, restore: list[str]) -> str:
    """Re-capitalise sentence-initial words, allowing for a leading '+'."""
    if not restore:
        return marked
    it = iter(restore)

    def sub(m: re.Match) -> str:
        try:
            next(it)
        except StopIteration:
            return m.group(0)
        tok = m.group(1)
        for i, ch in enumerate(tok):
            if ch != STRESS:
                return m.group(0)[: m.start(1) - m.start()] + \
                       tok[:i] + tok[i].upper() + tok[i + 1:]
        return m.group(0)

    pat = re.compile(r"(?:^|(?<=[.!?…])\s+|(?<=\n))\s*([+а-яё]+)")
    return pat.sub(sub, marked)


def _apply_phrase_overrides(marked: str) -> str:
    """Contextual corrections that a per-word dictionary cannot express.

    Matching is case-insensitive so patterns need not be written twice, so the
    original capitalisation has to be put back afterwards -- otherwise a
    sentence-initial "Отсек" comes out lower-cased.
    """
    def case_preserving(pattern: str, repl: str):
        def sub(m: re.Match) -> str:
            out = m.expand(repl)
            src = m.group(0)
            # Restore the case of the first alphabetic character.
            for i, ch in enumerate(src):
                if ch.isalpha():
                    for j, oc in enumerate(out):
                        if oc.isalpha():
                            return out[:j] + (oc.upper() if ch.isupper()
                                              else oc.lower()) + out[j + 1:]
                    break
            return out
        return sub

    for pattern, repl in PHRASE_OVERRIDES:
        marked = re.sub(pattern, case_preserving(pattern, repl), marked,
                        flags=re.IGNORECASE)
    return marked


def _apply_overrides(marked: str, overrides: dict[str, str]) -> str:
    """Force known-correct stress, whatever the model decided."""
    if not overrides:
        return marked

    def sub(m: re.Match) -> str:
        tok = m.group(0)
        bare = tok.replace(STRESS, "").lower()
        fixed = overrides.get(bare)
        if not fixed:
            return tok
        return fixed.capitalize() if tok[:1].isupper() else fixed

    return re.sub(r"[+А-Яа-яЁё]+", sub, marked)


def _tidy(s: str) -> str:
    """Normalise RUAccent output for Silero.

    Marks on single-vowel words are KEPT. An earlier version stripped them on
    the theory that Silero's stress_single_vowel covers that case -- it does not
    when put_accent=False, so "Дон" arrived with no stress information at all
    and the vowel came out reduced ("swallowed"). RUAccent marks it "Д+он";
    that mark has to survive.

    Only genuinely meaningless marks are dropped: one not followed by a vowel,
    and duplicates within a word.
    """
    s = re.sub(r"\+(?![аеёиоуыэюяАЕЁИОУЫЭЮЯ])", "", s)

    def per_word(m: re.Match) -> str:
        w = m.group(0)
        first = w.find(STRESS)
        if first == -1:
            return w
        bare = w.replace(STRESS, "")
        # A clitic with only one vowel takes no stress of its own, whatever the
        # model said. Multi-vowel words in the list (его, ему) keep theirs.
        if (bare.lower() in CLITICS
                and sum(c.lower() in VOWELS for c in bare) <= 1):
            return bare
        # keep only the first mark if the model emitted several
        return w[:first + 2] + w[first + 2:].replace(STRESS, "")

    return re.sub(r"[+А-Яа-яЁё]+", per_word, s)


def stress_positions(marked: str) -> dict[str, str]:
    """Map bare word -> marked form, for diffing two accentuations."""
    out: dict[str, str] = {}
    for tok in re.findall(r"[+А-Яа-яЁё]+", marked):
        bare = tok.replace(STRESS, "").lower()
        if bare:
            out[bare] = tok.lower()
    return out


# --------------------------------------------------------------------------
# alternate stress encoding for non-Silero models
# --------------------------------------------------------------------------

ACUTE = "\u0301"          # COMBINING ACUTE ACCENT


def to_acute(marked: str) -> str:
    """Convert Silero's "+vowel" marks to standard Russian "vowel + U+0301".

    Measured: feeding "+" marks to Chatterbox Multilingual destroys the output
    (58.5% CER -- it vocalises the plus signs). The combining acute, which is
    the ordinary orthographic way to write Russian stress and therefore appears
    in training data, is accepted cleanly at 0.0% CER.

    ё already carries stress inherently and takes no acute.
    """
    out: list[str] = []
    i = 0
    while i < len(marked):
        ch = marked[i]
        if ch == STRESS and i + 1 < len(marked):
            v = marked[i + 1]
            out.append(v)
            if v.lower() != "ё":
                out.append(ACUTE)
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)

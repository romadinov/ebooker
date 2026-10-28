"""Russian text -> StyleTTS2 phoneme string. Training and inference share this.

Two front-ends, chosen with ST2_FRONT (default "g2p"):

* **g2p**: silero ('+' stress, ё restored), then Russian-G2P's rule-based
  `word_to_ipa`, one word at a time. It applies Moscow-norm vowel reduction.
* **espeak**: silero, then sparse acute, then piper's espeak-ng. This was the
  first model's front-end. Kept only so that model's checkpoints can still be
  rendered.

Why g2p replaced espeak: listening to the espeak-trained model gave "an
impression that it is read by a non-native speaker", e.g. "случия" for
"случая". espeak's Russian does not reduce vowels the way a native speaker
does: случая -> sɭˈutʃʲʌja (native ˈsluʧɪjə), тяжёлым -> tʲaʒˈoɭym (tʲɪʐˈoɫɨm),
and every final unstressed -а is a full a. A phoneme-driven model can only be
as native as the phonemes it is given.

Russian-G2P (github.com/maxmelichov/Russian-G2P) declares no licence, so it is
not vendored. Point RU_G2P_PATH at a checkout.
"""
from __future__ import annotations

import os
import re
import sys

ACUTE = "́"
STRESS = "+"
VOWELS = "аеёиоуыэюя"

_state: dict = {}


def _acute_to_plus(word: str) -> str:
    """за́мок -> з+амок: the acute follows its vowel, '+' precedes it."""
    out = []
    for ch in word:
        if ch == ACUTE and out:
            out.insert(len(out) - 1, STRESS)
        else:
            out.append(ch)
    return "".join(out)


def _mark(text: str) -> str:
    """silero '+' marks. Words the caller already stressed by hand keep theirs."""
    if "acc" not in _state:
        from ebooker.ru_stress import SileroAccentuator
        _state["acc"] = SileroAccentuator()
    words = text.split(" ")
    bare = " ".join(w.replace(ACUTE, "") for w in words)
    marked = _state["acc"].mark(bare).split(" ")
    if len(marked) == len(words):
        marked = [_acute_to_plus(w) if ACUTE in w else m for w, m in zip(words, marked)]
    return " ".join(marked)


# Punctuation StyleTTS2's table has, and which carries a pause: keep it.
# Quotes carry none: drop them.
_PUNCT = {",": ",", ".": ".", "!": "!", "?": "?", ";": ";", ":": ":",
          "…": "…", "—": "—", "–": "—"}
_TOKEN = re.compile(r"[+а-яё]+(?:-[+а-яё]+)*|\.\.\.|[—–]|-(?=\s)|[,.!?;:…]", re.IGNORECASE)


def _g2p(text: str) -> str:
    if "w2i" not in _state:
        sys.path.insert(0, os.environ.get("RU_G2P_PATH", os.path.expanduser("~/Russian-G2P")))
        from data.russian_g2p_rules import word_to_ipa
        from data.russian_phonemizer import _expand_numbers
        _state["w2i"], _state["num"] = word_to_ipa, _expand_numbers
    w2i = _state["w2i"]
    out: list[str] = []
    # Numbers first: its expander agrees case with the next word ("в 1945 году"
    # -> "в тысяча девятьсот сорок пятом году"); text without digits is untouched.
    for tok in _TOKEN.findall(_mark(_state["num"](text)).lower()):
        if tok[0] in "+" + VOWELS or tok[0].isalpha():
            # Hyphenated words (кто-то, из-за) are one word per part. No ˈ on any
            # monosyllable: there it carries no information, and upstream's
            # habit of marking а/на/о/по/то would read as stressed prepositions.
            out.append(" ".join(w2i(p, mark_mono=False) for p in tok.split("-") if p.strip("+")))
        else:
            p = "…" if tok == "..." else _PUNCT.get(tok, "—")
            if out and p in ",.!?;:…":
                out[-1] += p       # attached, as in the training lists
            elif out:
                out.append(p)
    # ʦ/ʣ are outside StyleTTS2's 178-symbol table; spell them as t+s / d+z
    return " ".join(out).replace("ʦ", "ts").replace("ʣ", "dz").strip()


def _espeak(text: str) -> str:
    if "ph" not in _state:
        from ebooker.ru_stress import to_acute_sparse
        from piper.phonemize_espeak import EspeakPhonemizer
        _state["ph"], _state["sparse"] = EspeakPhonemizer(), to_acute_sparse
    marked = _state["sparse"](_mark(text).replace(STRESS, "+"))
    ipa = " ".join("".join(s) for s in _state["ph"].phonemize("ru", marked)).strip()
    # espeak spells soft р before a consonant "rɪ^" (дверью -> dvʲˈerɪ^jju), and
    # "^" is not in the symbol table. Elsewhere soft р is "rʲ".
    return ipa.replace("rɪ^", "rʲ")


def phonemise(text: str) -> str:
    front = os.environ.get("ST2_FRONT", "g2p")
    return {"g2p": _g2p, "espeak": _espeak}[front](text)

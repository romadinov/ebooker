"""Regression cases for the ASR round-trip checks.

Every case here came from listening to real output and finding either a defect
the checks missed or a correct rendering they wrongly flagged. The two halves
pull against each other -- loosening a rule to stop false alarms is what let
real drops through, and tightening it again is what produced the false alarms --
so they are kept together deliberately.

Run with:  uv run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from ebooker import verify as V


class _FixedTranscriber:
    """Stands in for Whisper: returns the transcript the case is testing."""

    def __init__(self, text: str):
        self.text = text

    def __call__(self, audio, sample_rate, lang):
        return self.text


def _flags(source: str, heard: str, lang: str, cps: float = 19.5) -> list[str]:
    """Run the real check, ignoring signal reasons a synthetic buffer triggers."""
    seconds = max(len(source) / cps, 0.4)
    audio = (np.random.default_rng(0).standard_normal(int(seconds * 24000))
             * 0.2).astype(np.float32)
    result = V.check(audio, 24000, source, lang,
                     transcriber=_FixedTranscriber(heard), cps=cps)
    return [r for r in result.reasons
            if not r.startswith(("repetition", "clipped"))]


# --- correct audio that must NOT be flagged --------------------------------
# Each of these was flagged at some point, sending a good chunk back for three
# pointless retries.
CLEAN = [
    # Russian numerals are gendered; num2words only yields the masculine, so a
    # source saying "две" against an ASR writing "2" looked like a drop. This
    # one mismatch produced 146 of 424 flags on a single book.
    ("ru", "обретали две тысячи дней веселья и вкусной еды тогда",
           "обретали 2 тысячи дней веселья и вкусной еды тогда"),
    ("ru", "одна из них обернулась ко мне и тихо улыбнулась мне",
           "1 из них обернулась ко мне и тихо улыбнулась мне"),
    # A three-word reply is about a second of audio; the ASR routinely drops the
    # dialogue dash, giving a word deficit and a 15% CER on correct audio.
    ("ru", "— Да, — подтвердила Мара.", "— Да, подтвердила Мара."),
    # Russian case endings differ by 2-3 characters, which defeats a tight
    # edit-distance budget: both words here were spoken.
    ("ru", "внешняя армированная обшивка смогла лишь немного смягчить его",
           "внешне армированная обшивка смогла лишь немного смягчить его"),
    # Whisper writes ordinals as digits with a suffix.
    ("ru", "После тридцатого дня полета все пассажиры ублажали себя так",
           "После 30-го дня полета все пассажиры ублажали себя так"),
    ("ru", "Он знал, что кто-то придет за ним очень скоро совсем",
           "Он знал, что кто то придет за ним очень скоро совсем"),
    # ASCII hyphens join rather than separate: the ASR returns these solid.
    ("en", "a fat, brat-and-beer-filled Chicago cop stood right there",
           "a fat, brat and beer filled Chicago cop stood right there"),
    # Thousands separators must not be read as a decimal.
    ("en", "But I lived in a town of two thousand people and everyone knew me",
           "But I lived in a town of 2,000 people and everyone knew me"),
    # A British voice makes the ASR transcribe in British spelling.
    ("en", "eating donuts at the Greenville Community Center that day",
           "eating doughnuts at the Greenville Community Centre that day"),
    # Clock times: "8.30am" must fold to the same words the model was given.
    ("en", "It departs at eight thirty in the morning we suggest you arrive",
           "It departs at 8.30am. We suggest you arrive"),
]

# --- real defects that MUST be flagged -------------------------------------
DEFECTIVE = [
    # Reported by a listener on a delivered book: a four-letter word vanished
    # from a 244-character chunk. Nothing flagged it -- at the old content-word
    # floor of 5 it was not even examined, and 4 letters missing from 244 is
    # about 1.6% CER. The underlying cause was chunk length (long chunks lose
    # short words, see normalise.MAX_CHUNK_CHARS); this case guards the
    # detection side.
    ("ru", "Армянское радио спрашивают: можно ли доехать на осле от Ташкента до Москвы?",
           "Армянское радио спрашивают: можно ли доехать на от Ташкента до Москвы?"),
    # A mangled numeral inside 149 characters scores only 5.4% CER, under any
    # sensible threshold -- which is why numerals are checked separately.
    ("ru", "поблизости оказавшиеся шестнадцать пассажиров тоже погибли и была часть",
           "поблизости оказавшиеся шить пассажиров тоже погибли и была часть"),
    # "борт" is four characters and was merged into the next word, leaving a
    # deficit of one and a CER lower than a legitimately clean render.
    ("ru", "Пассажиры поднимались на борт ракетного омнибуса Йоган Кеплер и обретали девяносто два дня.",
           "Пассажиры поднимались на бракетного омнибуса Йоганн Кеплер и обретали 92 дня."),
    # The gender fold must not hide an actually missing "две".
    ("ru", "прошло две минуты и всё стихло вокруг них совсем тогда",
           "прошло минуты и всё стихло вокруг них совсем тогда"),
    ("ru", "Пассажиры поднимались на борт ракетного омнибуса Йоган Кеплер и обретали девяносто два дня.",
           "Пассажиры поднимались на Йоганн Кеплер и обретали 92 дня."),
    ("en", "Your shuttle leaves in thirty minutes from right in front of this office",
           "Your shuttle leaves in minutes from right in front of this office"),
]


@pytest.mark.parametrize("lang,source,heard", CLEAN)
def test_correct_audio_is_not_flagged(lang, source, heard):
    assert _flags(source, heard, lang) == []


@pytest.mark.parametrize("lang,source,heard", DEFECTIVE)
def test_real_defects_are_flagged(lang, source, heard):
    assert _flags(source, heard, lang), "a real defect went undetected"


def test_stress_is_undetectable_by_asr():
    """The limit of this whole approach, asserted so it is not forgotten.

    Speech recognition returns word identity, not prosody, so за́мок and замо́к
    transcribe identically. A perfect CER says nothing about whether the stress
    was right; only listening does.
    """
    assert V.cer("замок", "замок", "ru") == 0.0
    assert _flags("старый замок на двери", "старый замок на двери", "ru") == []


def test_every_module_imports():
    """Guards against a broken module hiding behind lazy imports.

    A syntax error in normalise once passed 15 of 16 tests, because verify
    imports it lazily inside one function and no test exercised that path.
    """
    import importlib
    for name in ("ingest", "normalise", "verify", "master", "package",
                 "device", "ru_stress", "cli", "synth", "synth.base"):
        importlib.import_module(f"ebooker.{name}")


def test_bare_numeral_chunks_are_merged():
    """A chunk that is only a section number synthesises as silence."""
    from ebooker import normalise as N
    assert N._is_bare_numeral("четыре.", "ru")
    assert N._is_bare_numeral("five.", "en")
    assert not N._is_bare_numeral("– Жрет много?", "ru")
    assert not N._is_bare_numeral("четыре часа спустя", "ru")

    chunks, notes = N.chunk_paragraphs(
        ["четыре.", "Он вошел в комнату иñ огляделся вокруг очень внимательно."],
        "ru", 1)
    assert not any(N._is_bare_numeral(c.text, "ru") for c in chunks)
    assert any("numeral" in n for n in notes)


@pytest.mark.parametrize("audio,label", [
    (np.zeros(0, dtype=np.float32), "empty"),
    (np.zeros(1, dtype=np.float32), "one sample"),
    (np.array(0.0, dtype=np.float32), "zero-dimensional"),
    (np.zeros(240, dtype=np.float32), "10 ms"),
])
def test_degenerate_audio_flags_rather_than_raising(audio, label):
    """One bad chunk must not kill a run.

    The synthesis backend genuinely returns empty output for some inputs, and
    np.interp raised "object of too small depth for desired array" on it,
    taking down a whole book instead of letting the retry loop see a failed
    chunk. Nine chapters of a 102-chapter book were lost to that.
    """
    result = V.check(audio, 24000, "четыре.", "ru", transcriber=None)
    assert not result.ok, f"{label} should be flagged"
    assert result.reasons


def test_resample_handles_degenerate_input():
    assert V._resample(np.zeros(0, dtype=np.float32), 24000, 16000).shape == (0,)
    assert V._resample(np.array(0.0, dtype=np.float32), 24000, 16000).shape == (1,)

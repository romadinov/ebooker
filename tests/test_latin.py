"""Latin script in a Russian book: transliteration and routing.

The examples are the ones from chapter 8 of Б. Вавилонская, the satire that
slides from «Marlboro» into pages of broken English, which left 99 chunks
flagged when the Russian model simply skipped every Latin word.

Run with:  uv run pytest -q
"""

from __future__ import annotations

from ebooker import normalise
from ebooker.latin import ACUTE, is_english, latin_share, transliterate, transliterate_word


def bare(s: str) -> str:
    return s.replace(ACUTE, "")


def test_brand_names_follow_russian_practice():
    assert bare(transliterate_word("Coca")) == "Кока"
    assert bare(transliterate_word("Cola")) == "Кола"
    assert bare(transliterate_word("Marlboro")) == "Марлборо"
    assert bare(transliterate_word("Camel")) == "Кэмел"
    assert bare(transliterate_word("Pepsi")) == "Пепси"
    assert bare(transliterate_word("Cosmopolitan")) == "Козмополитан"


def test_sound_not_spelling():
    assert bare(transliterate_word("tough")) == "таф"
    assert bare(transliterate_word("Cherokee")) == "Чероки"
    assert bare(transliterate_word("jeeps")) == "джипс"
    assert bare(transliterate_word("way")) == "вей"


def test_reduced_vowels_follow_the_letters():
    assert bare(transliterate_word("residence")) == "резиденс"
    assert bare(transliterate_word("documents")) == "докюментс"
    assert bare(transliterate_word("president")) == "президент"


def test_primary_stress_becomes_an_acute():
    assert transliterate_word("Camel") == "Кэ" + ACUTE + "мел"
    assert transliterate_word("casino") == "каси" + ACUTE + "но"
    # one vowel: nothing to mark
    assert ACUTE not in transliterate_word("cash")


def test_suffixes_and_apostrophes():
    assert bare(transliterate_word("parkings")) == "паркингз"
    assert bare(transliterate_word("drinked")) == "дринкт"
    assert bare(transliterate_word("consensus'a")) == "консенсуса"
    assert bare(transliterate_word("it's")) == "итс"


def test_short_all_caps_is_spelt_out():
    assert transliterate_word("VIP") == "ви-ай-пи"
    assert transliterate_word("TV") == "ти-ви"


def test_roman_numerals_are_russian_ordinals():
    assert transliterate("в XX веке")[0] == "в двадцатом веке"
    assert transliterate("конец XIX века")[0] == "конец девятнадцатого века"
    assert transliterate("Пётр I")[0] == "Пётр первый"
    assert transliterate("XXI век")[0] == "двадцать первый век"
    assert transliterate("III")[0] == "третий"


def test_cyrillic_left_in_an_english_sentence_is_romanised():
    from ebooker.latin import romanize
    assert romanize("But he drinked водка too much.") == "But he drinked vodka too much."


def test_unknown_word_falls_back_to_spelling():
    assert bare(transliterate_word("Barbiedoll")) == "Барбиедолл"


def test_transliterate_reports_pairs_and_keeps_the_rest():
    out, pairs = transliterate("Ковбои из «Marlboro» прискакали на все фасады.")
    assert bare(out) == "Ковбои из «Марлборо» прискакали на все фасады."
    assert pairs == [("Marlboro", out.split("«")[1].split("»")[0])]


def test_english_sentence_detection():
    assert is_english("Gunmen shot on the streets!")
    # pidgin stays with the Russian narrator, accent and all
    assert not is_english("А президент drinked in residence.")
    assert not is_english("Простым people to жрать was nothing!")
    assert not is_english("And работал с documents in дача.")
    assert not is_english("Ковбои из «Marlboro» прискакали на все фасады.")
    assert latin_share("Только кириллица.") == 0.0


def test_chunks_never_mix_languages():
    paras = ["Ковбои из «Marlboro» прискакали. Gunmen shot on the streets! "
             "Gangster power takes victory! Racket процветал."]
    chunks, notes = normalise.chunk_paragraphs(paras, "ru", 8, max_chars=160)
    langs = [c.lang for c in chunks]
    assert langs == ["", "en", ""]
    assert "Марлборо" in bare(chunks[0].text)
    assert chunks[1].text == "Gunmen shot on the streets! Gangster power takes victory!"
    assert bare(chunks[2].text) == "Рэкет процветал."
    assert chunks[-1].is_paragraph_end and not chunks[0].is_paragraph_end
    assert any("routed" in n for n in notes) and any("transliterated" in n for n in notes)


def test_translit_mode_keeps_one_voice():
    paras = ["Gunmen shot on the streets! Racket процветал."]
    chunks, _ = normalise.chunk_paragraphs(paras, "ru", 8, max_chars=160, latin="translit")
    assert [c.lang for c in chunks] == [""]
    assert "Gunmen" not in chunks[0].text


def test_keep_mode_changes_nothing():
    paras = ["Gunmen shot on the streets! Racket процветал."]
    chunks, _ = normalise.chunk_paragraphs(paras, "ru", 8, max_chars=160, latin="keep")
    assert chunks[0].text == paras[0] and chunks[0].lang == ""


def test_hyphenated_compound_keeps_one_stress_mark():
    from ebooker.ru_stress import to_acute_sparse
    assert to_acute_sparse("К+ока-К+ола") == "Кока-Ко" + ACUTE + "ла"
    assert to_acute_sparse("Нью-Й+орке") == "Нью-Йо" + ACUTE + "рке"
    assert to_acute_sparse("кокт+ейль") == "кокте" + ACUTE + "йль"

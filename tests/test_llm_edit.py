from ebooker import llm_edit


def test_letters_ignores_marks_punctuation_and_yo():
    a = "Страсть к наживе погубит народ, надо быть добрыми."
    b = "Страсть к нажи́ве погу́бит наро́д; на́до быть до́брыми!"
    assert llm_edit.letters(a) == llm_edit.letters(b)
    assert llm_edit.letters("другой раздернул") == llm_edit.letters("друго́й раздёрнул")


def test_letters_catches_changed_words():
    assert llm_edit.letters("цвета крови") != llm_edit.letters("цвета кровь")


def test_scale_pause():
    assert llm_edit.scale_pause(700, None) == 700
    assert llm_edit.scale_pause(700, "long") == 1300
    assert llm_edit.scale_pause(350, "short") == 175
    assert llm_edit.scale_pause(100, "short") == 150

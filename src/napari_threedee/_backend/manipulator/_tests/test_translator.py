from napari_threedee._backend.manipulator.translator import TranslatorSet, Translator


def test_translator_instantiation():
    translator = Translator.from_string('x')
    assert isinstance(translator, Translator)


def test_translator_set_instantiation():
    translators = TranslatorSet.from_string('xyz')
    assert isinstance(translators, TranslatorSet)
    assert len(translators) == 3

"""TTS backends behind one interface, so the pipeline is model-agnostic."""

from .base import Backend, Result

__all__ = ["Backend", "Result", "get_backend", "default_backend", "BACKENDS"]

BACKENDS = ("silero", "kokoro", "espeech", "chatterbox", "qwen-stress")

# Chosen by measurement, not preference -- see DESIGN.md sections 6a and 6b.
#
# ru: Silero v5 scored 0.0% median CER against Chatterbox's 4.3%, resolves
#     stress and homographs natively, and runs 340x faster.
# en: every backend tested scored 0.0% CER, so cost decides; Kokoro is ~12x
#     faster than Chatterbox-Turbo and ~35x faster than Chatterbox base.
_BY_LANGUAGE = {
    "ru": "silero",
    "uk": "silero",
    "be": "silero",
    "en": "kokoro",
    "es": "kokoro",
    "fr": "kokoro",
    "it": "kokoro",
    "pt": "kokoro",
    "hi": "kokoro",
    "ja": "kokoro",
    "zh": "kokoro",
}


def default_backend(lang: str) -> str:
    """Backend for a language. Chatterbox is never the default: it is opt-in,
    for when one cloned voice across languages matters more than throughput."""
    return _BY_LANGUAGE.get((lang or "").lower().split("-")[0], "chatterbox")


def get_backend(name: str, **kw) -> Backend:
    if name == "silero":
        from .silero import Silero
        return Silero(**kw)
    if name == "kokoro":
        from .kokoro import Kokoro
        return Kokoro(**kw)
    if name == "espeech":
        from .espeech import ESpeech
        return ESpeech(**kw)
    if name == "chatterbox":
        from .chatterbox import Chatterbox
        return Chatterbox(**kw)
    if name == "qwen-stress":
        from .qwen_stress import QwenStress
        return QwenStress(**kw)
    if name == "omnivoice":
        # Absent from BACKENDS for the same reason as higgs: the model
        # repository declares no licence, so it is not published here.
        try:
            from .omnivoice import OmniVoice
        except ImportError as e:
            raise ValueError(
                "the 'omnivoice' backend is local-only and not part of this "
                "repository: the upstream model declares no licence. "
                f"Supply src/ebooker/synth/omnivoice.py yourself to use it ({e})."
            ) from e
        return OmniVoice(**kw)
    if name == "piper":
        # Local-only: piper-tts is GPL-3.0 and the ruslan voice CC BY-NC-SA.
        try:
            from .piper import Piper
        except ImportError as e:
            raise ValueError(
                "the 'piper' backend is local-only and not part of this "
                "repository: piper-tts is GPL-3.0 and the ru_RU-ruslan voice is "
                f"non-commercial. Supply src/ebooker/synth/piper.py yourself ({e})."
            ) from e
        return Piper(**kw)
    if name == "higgs":
        # Deliberately absent from BACKENDS: the Higgs weights are under a
        # research/non-commercial licence incompatible with this project's MIT
        # licence, so synth/higgs.py is not published. It stays reachable by
        # name for local use, with an explanatory failure if the file is absent.
        try:
            from .higgs import Higgs
        except ImportError as e:
            raise ValueError(
                "the 'higgs' backend is local-only and not part of this "
                "repository: its weights are research/non-commercial "
                "(Boson Higgs TTS 3 licence), which MIT cannot carry. "
                f"Supply src/ebooker/synth/higgs.py yourself to use it ({e})."
            ) from e
        return Higgs(**kw)
    raise ValueError(f"unknown backend: {name!r}, expected one of {BACKENDS}")

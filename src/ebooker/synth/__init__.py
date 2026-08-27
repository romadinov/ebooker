"""TTS backends behind one interface, so the pipeline is model-agnostic."""

from .base import Backend, Result

__all__ = ["Backend", "Result", "get_backend", "default_backend", "BACKENDS"]

BACKENDS = ("silero", "kokoro", "espeech", "chatterbox")

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
    raise ValueError(f"unknown backend: {name!r}, expected one of {BACKENDS}")

"""Per-chunk delivery tags for backends that accept inline emotion control.

Why this exists: a constant tag produces plain narration. Measured on the same
passage, Whisper recovered 13 of 11 questions and exclamations from audio
rendered with an emotion tag matched to the content, and 4 without one -- and 0
from the flow-matching backend at any setting. The expressiveness came from the
emotion tag, not from the model alone. Rendering a whole book under one tag
throws that away, which is what made a full-book sample read as flat despite
the model being capable of far more.

This is a first pass and deliberately shallow: it reads punctuation and
dialogue markers, not meaning. It cannot tell an angry question from a curious
one. A model annotating each sentence would do better and is the obvious next
step; this exists to establish whether varying the tag at all is audible,
without spending a book's worth of GPU to find out.

Only Higgs accepts these tags. ESpeech and Qwen ignore or vocalise them, so
callers must pass tags=None for those backends.
"""

from __future__ import annotations

import re

# Sentence-level tags from the Higgs catalog (21 emotions, 3 styles, 10 prosody).
BASELINE = "<|prosody:expressive_high|>"

# Ordered: the first matching rule wins, so put the strongest signals first.
_RULES: list[tuple[str, str]] = [
    # Shouted or emphatic: multiple marks, or an exclamation inside dialogue.
    (r"[!?]{2,}", "<|emotion:anger|>"),
    (r"^\s*[—–-].*!", "<|emotion:enthusiasm|>"),
    # A bare exclamation in narration is usually emphasis, not rage.
    (r"!", "<|emotion:determination|>"),
    # Questions: rhetorical asides in this corpus read wry rather than puzzled.
    (r"\?.*\?", "<|emotion:amusement|>"),
    (r"\?", "<|emotion:confusion|>"),
    # Dialogue without terminal punctuation cues: conversational.
    (r"^\s*[—–-]\s", "<|emotion:amusement|>"),
    # Quoted speech or an aside in guillemets: reported, slightly detached.
    (r"[«»\"]", "<|emotion:contemplation|>"),
]

# Narration default. Not neutral on purpose: "contemplation" gives the reading
# a settled, unhurried contour, where no tag at all comes out inert.
_NARRATION = "<|emotion:contemplation|>"


def tags_for(text: str, *, baseline: str = BASELINE) -> str:
    """Delivery tags to prefix to ``text``. Sentence-level, so they lead."""
    for pattern, tag in _RULES:
        if re.search(pattern, text, re.MULTILINE):
            return baseline + tag
    return baseline + _NARRATION


def summarise(texts: list[str]) -> dict[str, int]:
    """Tag distribution over a list of chunks, for sanity-checking a book."""
    counts: dict[str, int] = {}
    for t in texts:
        tag = tags_for(t).replace(BASELINE, "") or "(baseline only)"
        counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

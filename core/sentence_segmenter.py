"""
Sentence segmentation helper for Michael GPT-2.
Falls back to the old regex if NLTK Punkt is not available.
"""
from __future__ import annotations

import re
try:
    from nltk.tokenize import sent_tokenize as _punkt
    _have_punkt = True
except ModuleNotFoundError:           # NLTK not installed
    _have_punkt = False

_REGEX = re.compile(r'(?<=[.!?])\s+')

def segment_text(text: str) -> list[str]:
    """
    Return a list of sentences, using NLTK Punkt when present,
    otherwise the legacy regex splitter.
    """
    if _have_punkt:
        return _punkt(text)
    return _REGEX.split(text)


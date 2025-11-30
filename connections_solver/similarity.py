# connections_solver/similarity.py

from __future__ import annotations
from typing import Optional
import math
import logging

logger = logging.getLogger(__name__)


class SimilarityBackend:
    """
    Abstract similarity backend.

    Implementations must define similarity(w1, w2) -> float in [0, 1].
    """
    def similarity(self, w1: str, w2: str) -> float:
        raise NotImplementedError


class SpacyBackend(SimilarityBackend):
    """
    Uses spaCy word vectors if installed.
    """

    def __init__(self, model_name: str = "en_core_web_md"):
        import spacy  # type: ignore
        self.nlp = spacy.load(model_name)

    def similarity(self, w1: str, w2: str) -> float:
        d1 = self.nlp(w1.replace("_", " "))
        d2 = self.nlp(w2.replace("_", " "))
        # spaCy similarity is roughly in [0, 1], but can vary.
        return float(d1.similarity(d2))


class CharacterNgramBackend(SimilarityBackend):
    """
    Cheap fallback similarity based on character n-gram Jaccard overlap.
    """

    def __init__(self, n: int = 2):
        self.n = n

    def _ngrams(self, s: str):
        s = s.lower()
        return {s[i:i + self.n] for i in range(max(len(s) - self.n + 1, 1))}

    def similarity(self, w1: str, w2: str) -> float:
        n1 = self._ngrams(w1)
        n2 = self._ngrams(w2)
        inter = len(n1 & n2)
        union = len(n1 | n2)
        if union == 0:
            return 0.0
        return inter / union


def get_default_backend() -> SimilarityBackend:
    """
    Try to build a spaCy-based backend. If that fails, fall back to character n-grams.
    """
    try:
        backend = SpacyBackend()
        logger.info("Using spaCy similarity backend.")
        return backend
    except Exception as e:
        logger.warning("Falling back to CharacterNgramBackend: %s", e)
        return CharacterNgramBackend()

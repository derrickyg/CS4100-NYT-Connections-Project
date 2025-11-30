# scoring.py

from __future__ import annotations
from typing import Iterable, List
import itertools

from .similarity import SimilarityBackend


def group_cohesion(group: Iterable[str], backend: SimilarityBackend) -> float:
    """
    Average pairwise similarity within a group. Higher is better.
    """
    words = list(group)
    if len(words) < 2:
        return 0.0

    total = 0.0
    count = 0
    for w1, w2 in itertools.combinations(words, 2):
        total += backend.similarity(w1, w2)
        count += 1
    return total / count if count else 0.0


def partition_score(groups: List[Iterable[str]], backend: SimilarityBackend) -> float:
    """
    Score a full partition (4 groups) as the sum of group cohesions.

    """
    return sum(group_cohesion(group, backend) for group in groups)

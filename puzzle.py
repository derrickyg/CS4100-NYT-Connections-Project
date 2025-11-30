# connections_solver/puzzle.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Iterable, Set
import json
from pathlib import Path


@dataclass(frozen=True)
class Puzzle:
    """
    Represents a single Connections puzzle.

    Attributes
    ----------
    words : List[str]
        The 16 words shown in the grid.
    solution_groups : List[Set[str]]
        The true 4 solution groups (sets of 4 words each), if known.
        This is only used for evaluation / grading, not by the solver itself.
    labels : List[str]
        Optional category labels in the same order as solution_groups.
    """
    words: List[str]
    solution_groups: List[Set[str]]
    labels: List[str]

    @classmethod
    def from_json(cls, path: str | Path) -> "Puzzle":
        """
        Load a puzzle from a JSON file.

        Expected format:
        {
            "words": ["WORD1", ..., "WORD16"],
            "groups": [
                {"label": "Category 1", "words": ["WORDa", "WORDb", "WORDc", "WORDd"]},
                ...
            ]
        }
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        words = data["words"]
        groups = data.get("groups", [])
        solution_groups: List[Set[str]] = []
        labels: List[str] = []

        for g in groups:
            labels.append(g.get("label", ""))
            solution_groups.append(set(g["words"]))

        return cls(words=words, solution_groups=solution_groups, labels=labels)

    def evaluate_partition(self, groups: List[Iterable[str]]) -> Dict[str, float]:
        """
        Compare a candidate partition to the true solution_groups.

        Returns a dict with simple accuracy metrics:
        - group_matches: number of groups that exactly match a true group
        - word_correct_group: number of words assigned to the correct group
        """
        if not self.solution_groups:
            raise ValueError("No solution_groups available for evaluation.")

        candidate = [set(g) for g in groups]
        true_groups = self.solution_groups

        # Exact group matches
        group_matches = sum(1 for g in candidate if g in true_groups)

        # Word-level correctness
        word_correct = 0
        total_words = len(self.words)
        word_to_true_group_idx: Dict[str, int] = {}
        for idx, g in enumerate(true_groups):
            for w in g:
                word_to_true_group_idx[w] = idx

        for g_idx, g in enumerate(candidate):
            for w in g:
                if w in word_to_true_group_idx and word_to_true_group_idx[w] == g_idx:
                    word_correct += 1

        return {
            "group_matches": float(group_matches),
            "word_correct_group": float(word_correct),
            "word_accuracy": float(word_correct) / total_words if total_words else 0.0,
        }

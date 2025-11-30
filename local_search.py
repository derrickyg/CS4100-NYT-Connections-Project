# local_search.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random
import math

from .similarity import SimilarityBackend
from .scoring import partition_score


@dataclass
class SearchResult:
    groups: List[List[str]]
    score: float
    iterations: int
    restarts: int


def random_partition(words: List[str], group_size: int = 4) -> List[List[str]]:
    shuffled = words[:]
    random.shuffle(shuffled)
    return [shuffled[i:i + group_size] for i in range(0, len(shuffled), group_size)]


def neighbors(partition: List[List[str]]) -> List[List[List[str]]]:
    """
    Generate neighbors by swapping two words in different groups.
    """
    neighbors_list: List[List[List[str]]] = []
    num_groups = len(partition)
    # Work with indices so we don't accidentally alias lists
    for i in range(num_groups):
        for j in range(i + 1, num_groups):
            for idx_i, w_i in enumerate(partition[i]):
                for idx_j, w_j in enumerate(partition[j]):
                    new_part = [g[:] for g in partition]
                    new_part[i][idx_i], new_part[j][idx_j] = new_part[j][idx_j], new_part[i][idx_i]
                    neighbors_list.append(new_part)
    return neighbors_list


def hill_climbing(
    words: List[str],
    backend: SimilarityBackend,
    max_iterations: int = 1000,
    restarts: int = 50,
) -> SearchResult:
    """
    Pure hill-climbing with random restarts.

    This is a local search algorithm: we only keep the best neighbor
    in the immediate neighborhood and restart if we get stuck in a local maximum.
    """
    best_overall: SearchResult | None = None
    total_iters = 0

    for r in range(restarts):
        current = random_partition(words)
        current_score = partition_score(current, backend)

        improved = True
        iters = 0

        while improved and iters < max_iterations:
            iters += 1
            total_iters += 1
            improved = False

            best_neighbor = current
            best_neighbor_score = current_score

            for nb in neighbors(current):
                s = partition_score(nb, backend)
                if s > best_neighbor_score:
                    best_neighbor = nb
                    best_neighbor_score = s

            if best_neighbor_score > current_score:
                current = best_neighbor
                current_score = best_neighbor_score
                improved = True

        result = SearchResult(groups=current, score=current_score,
                              iterations=total_iters, restarts=r + 1)

        if best_overall is None or result.score > best_overall.score:
            best_overall = result

    assert best_overall is not None
    return best_overall


def simulated_annealing(
    words: List[str],
    backend: SimilarityBackend,
    max_iterations: int = 5000,
    start_temperature: float = 1.0,
    cooling_rate: float = 0.0005,
) -> SearchResult:
    """
    Simulated annealing variant to escape local optima.

    We occasionally accept worse moves with probability depending
    on the temperature schedule.
    """
    current = random_partition(words)
    current_score = partition_score(current, backend)

    best = current
    best_score = current_score

    temp = start_temperature

    for it in range(max_iterations):
        temp = max(temp * (1.0 - cooling_rate), 1e-6)

        # propose a random neighbor
        nb_list = neighbors(current)
        candidate = random.choice(nb_list)
        candidate_score = partition_score(candidate, backend)

        delta = candidate_score - current_score
        if delta > 0:
            accept = True
        else:
            accept_prob = math.exp(delta / temp)
            accept = random.random() < accept_prob

        if accept:
            current = candidate
            current_score = candidate_score

            if current_score > best_score:
                best, best_score = current, current_score

    return SearchResult(groups=best, score=best_score,
                        iterations=max_iterations, restarts=1)

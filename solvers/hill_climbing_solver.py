import random
from typing import List, Dict, Tuple

from similarity.embedding_similarity import EmbeddingSimilarity


class HillClimbingConnectionsSolver:
    """
    Hill-climbing solver for NYT Connections.

    Given the 16 words of a puzzle, this solver searches over partitions of the
    words into 4 groups of 4. The objective is to maximize the sum of pairwise
    similarities within each group, using an EmbeddingSimilarity function.
    Similar to the K-Means solver in outputting 4 groups at once and then Game Simulator
    evaluates how the grouping would perform as a player.
    """


    def __init__(
        self,
        similarity_fn: EmbeddingSimilarity,
        max_restarts: int = 20,
        max_steps: int = 200,
    ):
        """
        Args:
            similarity_fn: Function object with signature
                similarity_fn.similarity(word1, word2) -> float in [0, 1].
            max_restarts: How many random restarts to use to escape local maxima
            max_steps: Maximum hill-climbing steps per restart
        """
        self.similarity_fn = similarity_fn
        self.max_restarts = max_restarts
        self.max_steps = max_steps


    
    def solve_constrained(self, words: List[str]) -> List[List[str]]:
        """
        Partition 16 words into 4 groups of 4 using hill climbing.

        Args:
            words: List of 16 puzzle words.

        Returns:
            A list of 4 groups, where each group is a list of 4 words.
        """
        if len(words) != 16:
            raise ValueError(
                f"HillClimbingConnectionsSolver expects exactly 16 words, got {len(words)}"
            )

        # Precompute similarity matrix for speed
        sim_matrix, index_lookup = self._compute_similarity_matrix(words)

        best_global_groups = None
        best_global_score = float("-inf")

        for _ in range(self.max_restarts):
            # Start from a random valid partition
            current_groups = self._initial_partition(words)
            current_score = self._score_groups(current_groups, sim_matrix, index_lookup)

            for _ in range(self.max_steps):
                improved, new_groups, new_score = self._best_neighbor(
                    current_groups, current_score, sim_matrix, index_lookup
                )
                if not improved:
                    break  # local maximum reached
                current_groups, current_score = new_groups, new_score

            if current_score > best_global_score:
                best_global_score = current_score
                best_global_groups = current_groups

        # Optionally sort groups by cohesion 
        if best_global_groups is None:
            # Fallback to regular chunking
            shuffled = words[:]
            random.shuffle(shuffled)
            best_global_groups = [shuffled[i * 4 : (i + 1) * 4] for i in range(4)]

        return self._sort_groups_by_cohesion(best_global_groups, sim_matrix, index_lookup)


    def _compute_similarity_matrix(
        self, words: List[str]
    ) -> Tuple[List[List[float]], Dict[str, int]]:
        """
        Precompute pairwise similarities between all words.

        Returns:
            sim_matrix: 2D list where sim_matrix[i][j] is similarity between words[i] and words[j].
            index_lookup: dict mapping word -> its index in the words list.
        """
        n = len(words)
        sim_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        index_lookup = {w: i for i, w in enumerate(words)}

        for i in range(n):
            sim_matrix[i][i] = 1.0
            for j in range(i + 1, n):
                sim = self.similarity_fn.similarity(words[i], words[j])
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim

        return sim_matrix, index_lookup

    def _initial_partition(self, words: List[str]) -> List[List[str]]:
        """
        Create a random partition of the 16 words into 4 groups of 4.
        """
        shuffled = words[:]
        random.shuffle(shuffled)
        return [shuffled[i * 4 : (i + 1) * 4] for i in range(4)]

    def _score_groups(
        self,
        groups: List[List[str]],
        sim_matrix: List[List[float]],
        index_lookup: Dict[str, int],
    ) -> float:
        """
        Compute the total within-group similarity score.

        For each group, we sum the pairwise similarity of all word pairs inside
        that group. Higher scores mean more semantically cohesive clusters.
        """
        total_score = 0.0

        for group in groups:
            if len(group) < 2:
                continue
            indices = [index_lookup[w] for w in group]
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    total_score += sim_matrix[indices[i]][indices[j]]

        return total_score

    def _best_neighbor(
        self,
        groups: List[List[str]],
        current_score: float,
        sim_matrix: List[List[float]],
        index_lookup: Dict[str, int],
    ) -> Tuple[bool, List[List[str]], float]:
        """
        Find the best neighbor by swapping any pair of words from different groups.

        Returns:
            (improved, best_groups, best_score)
        """
        best_score = current_score
        best_groups = groups
        improved = False

        # There are 4 groups; groups[i] always has 4 words.
        num_groups = len(groups)

        for gi in range(num_groups):
            for gj in range(gi + 1, num_groups):
                group_i = groups[gi]
                group_j = groups[gj]
                for idx_i in range(len(group_i)):
                    for idx_j in range(len(group_j)):
                        # Make a candidate by swapping one word between group_i and group_j
                        candidate_groups = [list(g) for g in groups]  # shallow copy each group
                        candidate_groups[gi][idx_i], candidate_groups[gj][idx_j] = (
                            candidate_groups[gj][idx_j],
                            candidate_groups[gi][idx_i],
                        )

                        candidate_score = self._score_groups(
                            candidate_groups, sim_matrix, index_lookup
                        )

                        if candidate_score > best_score + 1e-9:
                            best_score = candidate_score
                            best_groups = candidate_groups
                            improved = True

        return improved, best_groups, best_score

    def _sort_groups_by_cohesion(
        self,
        groups: List[List[str]],
        sim_matrix: List[List[float]],
        index_lookup: Dict[str, int],
    ) -> List[List[str]]:
        """
        Sort groups by average internal similarity (most cohesive first).
        """
        def cohesion(group: List[str]) -> float:
            if len(group) < 2:
                return 0.0
            indices = [index_lookup[w] for w in group]
            num_pairs = 0
            score = 0.0
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    score += sim_matrix[indices[i]][indices[j]]
                    num_pairs += 1
            return score / num_pairs if num_pairs > 0 else 0.0

        return sorted(groups, key=cohesion, reverse=True)

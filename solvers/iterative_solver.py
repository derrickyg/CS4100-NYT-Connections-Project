"""
Iterative solver that uses game feedback to improve guesses.
"""
from typing import List, Dict, Optional, Set
from itertools import combinations
from similarity.embedding_similarity import EmbeddingSimilarity
from evaluation.game_simulator import GameSimulator, GameFeedback
import random
import config


class IterativeSolver:
    """Solver that submits groups iteratively using game feedback."""
    
    def __init__(self, similarity_function: EmbeddingSimilarity):
        """
        Initialize iterative solver.
        
        Args:
            similarity_function: Function to compute word similarity
        """
        self.similarity_fn = similarity_function
        # Track which words we know belong together
        self.known_pairs: Set[tuple] = set()
        # Track which words we know don't belong together
        self.forbidden_pairs: Set[tuple] = set()
        # Track submission history
        self.submission_history: List[Dict] = []
        # Track groups we've already tried (to avoid repeats)
        self.tried_groups: Set[tuple] = set()
    
    def _normalize_group(self, group: List[str]) -> tuple:
        """
        sort and uppercase the words in the group for comparison purposes
        
        Args:
            group: List of words
            
        Returns:
            Normalized tuple
        """
        return tuple(sorted(w.upper() for w in group))
    
    def _to_upper_set(self, words: List[str]) -> Set[str]:
        """Convert list of words to uppercase set."""
        return set(w.upper() for w in words)
    
    def _get_available_from_previous(self, previous_group: List[str], remaining_words: List[str]) -> List[str]:
        """Get words from previous group that are still available."""
        remaining_set = self._to_upper_set(remaining_words)
        return [w for w in previous_group if w.upper() in remaining_set]
    
    def _is_group_tried(self, group: List[str], tried_groups: Set[tuple]) -> bool:
        """Check if a group has already been tried."""
        return self._normalize_group(group) in tried_groups
    
    def solve_with_feedback(self, game: GameSimulator) -> Dict:
        """
        Solve puzzle by submitting groups iteratively and using feedback.
        
        Args:
            game: GameSimulator instance
            
        Returns:
            Dictionary with solution and statistics
        """
        submissions = []
        # Reset tried groups for each new puzzle
        self.tried_groups.clear()
        self.known_pairs.clear()
        self.forbidden_pairs.clear()
        self.submission_history.clear()
        
        while not game.is_game_over:
            remaining = game.get_remaining_words()
            if len(remaining) == 0:
                break
            
            # Select and submit next group (retry if already tried)
            max_attempts = 100
            group = None
            for _ in range(max_attempts):
                candidate = self._select_next_group(remaining, game, self.tried_groups)
                group_tuple = self._normalize_group(candidate)
                if group_tuple not in self.tried_groups:
                    group = candidate
                    self.tried_groups.add(group_tuple)
                    break
            
            # Fallback if all attempts failed - just get a random group
            if group is None:
                group = self._get_random_untried_group(remaining, self.tried_groups)
                if group:
                    group_tuple = self._normalize_group(group)
                    self.tried_groups.add(group_tuple)
            
            # Submit group
            submission_num = len(submissions) + 1
            feedback = game.submit_group(group)
            submission_data = {
                "group": group,
                "feedback": feedback
            }
            submissions.append(submission_data)
            self.submission_history.append(submission_data)
            
            # Learn from feedback
            self._learn_from_feedback(group, feedback, game)
            
            # Print correct guesses
            if feedback.is_correct:
                print(f"Correct guess #{submission_num}: {', '.join(group)}")
        
        # Get final state
        state = game.get_state()
        
        return {
            "solved_groups": game.get_solved_groups(),
            "submissions": submissions,
            "total_submissions": len(submissions),
            "mistakes": state["mistakes"],
            "is_won": state["is_won"]
        }
    
    def _select_next_group(self, remaining_words: List[str], game: GameSimulator, 
                          tried_groups: Set[tuple]) -> List[str]:
        """
        Select the next group of 4 words to submit.
        
        Args:
            remaining_words: Words not yet solved
            game: Game simulator
            tried_groups: Set of already-tried group tuples (to avoid repeats)
            
        Returns:
            List of 4 words to submit
        """
        if len(remaining_words) < 4:
            return remaining_words
        
        # Strategy 1: deal with partial matches (3/4 or 2/4 correct)
        for correct_count in [3, 2]:
            history_size = 10 if correct_count == 3 else 5
            for submission in reversed(self.submission_history[-history_size:]):
                if (not submission['feedback'].is_correct and 
                    submission['feedback'].correct_words == correct_count):
                    if correct_count == 3:
                        group = self._refine_partial_match(submission['group'], remaining_words, tried_groups)
                    else:
                        group = self._refine_two_correct(submission['group'], remaining_words, tried_groups)
                    
                    if group:
                        return group
        
        # Strategy 2: Build from known pairs
        if self.known_pairs:
            group = self._build_group_from_pairs(remaining_words, tried_groups)
            if group:
                return group
        
        # Strategy 3: Find most similar group
        group = self._find_most_similar_group(remaining_words, tried_groups)
        if group:
            return group
        
        # Fallback: random untried combination
        return self._get_random_untried_group(remaining_words, tried_groups)
    
    def _get_random_untried_group(self, remaining_words: List[str], tried_groups: Set[tuple]) -> List[str]:
        """Get a random group that hasn't been tried yet."""
        remaining_list = list(remaining_words)
        for _ in range(20):
            candidate = random.sample(remaining_list, 4)
            if not self._is_group_tried(candidate, tried_groups):
                return candidate
        
        # Last resort: return first 4 words
        return remaining_list[:4]
    
    def _refine_partial_match(self, previous_group: List[str], remaining_words: List[str], 
                            tried_groups: Set[tuple]) -> Optional[List[str]]:
        """
        Try to refine a group that had 3/4 correct by swapping the wrong word.
        
        Args:
            previous_group: Previous submission that had 3/4 correct
            remaining_words: Words still available
            
        Returns:
            Refined group of exactly 4 words or None
        """
        available_from_prev = self._get_available_from_previous(previous_group, remaining_words)
        
        if len(available_from_prev) < 3:
            return None
        
        # Try all combinations of 3 words from previous group
        best_group = None
        best_score = float('-inf')
        
        for three_words in combinations(available_from_prev, 3):
            three_words_set = self._to_upper_set(three_words)
            candidates = [w for w in remaining_words if w.upper() not in three_words_set]
            
            # Find best completion for this combination
            group = self._find_best_completion(list(three_words), candidates, tried_groups)
            if group:
                # Score this group to compare with others
                score = sum(
                    self.similarity_fn.similarity(w1, w2)
                    for w1, w2 in combinations(group, 2)
                )
                if score > best_score:
                    best_score = score
                    best_group = group
        
        return best_group
    
    def _refine_two_correct(self, previous_group: List[str], remaining_words: List[str], 
                           tried_groups: Set[tuple]) -> Optional[List[str]]:
        """
        When we got 2/4 correct, try different combinations of those 2 words.
        
        Args:
            previous_group: Previous submission with 2/4 correct
            remaining_words: Words still available
            
        Returns:
            New group or None
        """
        available_from_prev = self._get_available_from_previous(previous_group, remaining_words)
        
        if len(available_from_prev) < 2:
            return None
        
        # Try all pairs from the previous group
        for pair in combinations(available_from_prev, 2):
            pair_set = self._to_upper_set(pair)
            candidates = [w for w in remaining_words if w.upper() not in pair_set]
            
            if len(candidates) < 2:
                continue
            
            # Find best combination of 2 more words
            best_group = self._find_best_completion(pair, candidates, tried_groups)
            if best_group:
                return best_group
        
        return None
    
    def _find_best_completion(self, base_words: List[str], candidates: List[str], 
                             tried_groups: Set[tuple]) -> Optional[List[str]]:
        """
        Find the best 2-word completion for a base group of words.
        
        Args:
            base_words: Base words (2 words for 2/4 correct, 3 words for 3/4 correct)
            candidates: Candidate words to complete the group
            tried_groups: Set of already-tried groups
            
        Returns:
            Best group of 4 words or None
        """
        best_group = None
        best_score = float('-inf')
        words_needed = 4 - len(base_words)
        
        for completion in combinations(candidates, words_needed):
            test_group = list(base_words) + list(completion)
            
            if self._is_group_tried(test_group, tried_groups):
                continue
            
            # Score by sum of all pairwise similarities within the group
            score = sum(
                self.similarity_fn.similarity(w1, w2)
                for w1, w2 in combinations(test_group, 2)
            )
            
            if score > best_score:
                best_score = score
                best_group = test_group
        
        return best_group
    
    def _build_group_from_pairs(self, remaining_words: List[str], tried_groups: Set[tuple]) -> Optional[List[str]]:
        """
        Try to build a group from known pairs.
        
        Args:
            remaining_words: Available words
            tried_groups: Set of already-tried groups
            
        Returns:
            Group of 4 words or None
        """
        remaining_set = self._to_upper_set(remaining_words)
        
        # Find pairs where both words are still available
        available_pairs = [
            (w1, w2) for w1, w2 in self.known_pairs
            if w1.upper() in remaining_set and w2.upper() in remaining_set
        ]
        
        if not available_pairs:
            return None
        
        # Try each available pair to find the best completion
        for w1, w2 in available_pairs:
            pair = [w1, w2]
            pair_set = self._to_upper_set(pair)
            candidates = [w for w in remaining_words if w.upper() not in pair_set]
            
            if len(candidates) < 2:
                continue
            
            # Find best 2-word completion for this pair
            best_group = self._find_best_completion(pair, candidates, tried_groups)
            if best_group:
                return best_group
        
        return None
    
    def _find_most_similar_group(self, remaining_words: List[str], tried_groups: Set[tuple]) -> List[str]:
        """
        Find the group of 4 words with highest similarity.
        
        Args:
            remaining_words: Available words
            tried_groups: Set of already-tried groups
            
        Returns:
            Group of 4 words
        """
        if len(remaining_words) == 4:
            return remaining_words
        
        # Get combinations to try (prioritize those with known pairs)
        all_combos = self._get_prioritized_combinations(remaining_words)
        
        # Find best group from combinations
        best_group = None
        best_score = float('-inf')
        max_to_check = min(len(all_combos), config.ITERATIVE_MAX_TO_CHECK)
        
        for combo in all_combos[:max_to_check]:
            if self._is_group_tried(list(combo), tried_groups):
                continue
            
            if self._has_forbidden_pair(combo):
                continue
            
            score = self._score_group(combo)
            
            if score > best_score:
                best_score = score
                best_group = list(combo)
        
        if best_group:
            return best_group
        
        # Fallback: random untried combination
        return self._get_random_untried_group(remaining_words, tried_groups)
    
    def _get_prioritized_combinations(self, remaining_words: List[str]) -> List[tuple]:
        """
        Get combinations of 4 words, prioritizing those with known pairs.
        
        Args:
            remaining_words: Available words
            
        Returns:
            List of combinations (tuples of 4 words)
        """
        all_combos = list(combinations(remaining_words, 4))
        max_combinations = config.ITERATIVE_MAX_COMBINATIONS
        
        if len(all_combos) <= max_combinations:
            return all_combos
        
        # Split into combos with/without known pairs
        combos_with_pairs = []
        combos_without_pairs = []
        
        for combo in all_combos:
            combo_set = self._to_upper_set(list(combo))
            has_pair = any(
                w1.upper() in combo_set and w2.upper() in combo_set
                for w1, w2 in self.known_pairs
            )
            
            if has_pair:
                combos_with_pairs.append(combo)
            else:
                combos_without_pairs.append(combo)
        
        # Sample more from combos with known pairs
        half_max = max_combinations // 2
        prioritized = random.sample(combos_with_pairs, min(half_max, len(combos_with_pairs)))
        prioritized.extend(random.sample(combos_without_pairs, max_combinations - len(prioritized)))
        
        return prioritized
    
    def _has_forbidden_pair(self, combo: tuple) -> bool:
        """Check if combo contains any forbidden pair."""
        combo_set = self._to_upper_set(list(combo))
        return any(
            w1.upper() in combo_set and w2.upper() in combo_set
            for w1, w2 in self.forbidden_pairs
        )
    
    def _score_group(self, combo: tuple) -> float:
        """
        Score a group by similarity and known pairs bonus.
        
        Args:
            combo: Tuple of words in the group
            
        Returns:
            Score (higher is better)
        """
        # Sum of all pairwise similarities
        score = sum(
            self.similarity_fn.similarity(w1, w2)
            for w1, w2 in combinations(combo, 2)
        )
        
        # Bonus for known pairs
        combo_set = self._to_upper_set(list(combo))
        pair_bonus = sum(
            config.ITERATIVE_KNOWN_PAIR_BONUS
            for w1, w2 in self.known_pairs
            if w1.upper() in combo_set and w2.upper() in combo_set
        )
        
        return score + pair_bonus
    
    def _learn_from_feedback(self, group: List[str], feedback: GameFeedback, game: GameSimulator):
        """
        Learn from feedback to improve future guesses.
        
        Args:
            group: Submitted group
            feedback: Feedback received
            game: Game simulator
        """
        group_upper = [w.upper() for w in group]
        all_pairs = list(combinations(group_upper, 2))
        
        if feedback.is_correct:
            # All words in this group belong together
            self.known_pairs.update(all_pairs)
        elif feedback.correct_words == 0:
            # None of these words belong together
            self.forbidden_pairs.update(all_pairs)
        elif feedback.correct_words >= 2:
            # Some words belong together - identify likely pairs
            self._learn_from_partial_feedback(group_upper, all_pairs, feedback.correct_words)
    
    def _learn_from_partial_feedback(self, group_upper: List[str], all_pairs: List[tuple], correct_words: int):
        """
        Learn from partial feedback (2/4 or 3/4 correct).
        
        Args:
            group_upper: Uppercased words from the group
            all_pairs: All pairs from the group
            correct_words: Number of correct words (2 or 3)
        """
        # Score all pairs by similarity
        pair_scores = [
            (pair, self.similarity_fn.similarity(pair[0], pair[1]))
            for pair in all_pairs
        ]
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Number of pairs we expect: C(correct_words, 2)
        num_correct_pairs = correct_words * (correct_words - 1) // 2
        num_pairs_to_protect = min(num_correct_pairs, len(pair_scores))
        protected_pairs = {pair for pair, _ in pair_scores[:num_pairs_to_protect]}
        
        # Add top pairs as known
        self.known_pairs.update(protected_pairs)
        
        # For 3/4 matches, find the best trio and learn from it
        if correct_words == 3:
            best_trio = self._find_best_trio(group_upper)
            if best_trio:
                trio_pairs = list(combinations(best_trio, 2))
                self.known_pairs.update(trio_pairs)
                
                # Mark pairs involving the 4th word as forbidden
                bottom_word = [w for w in group_upper if w not in best_trio][0]
                forbidden_pairs = [(bottom_word, w) for w in best_trio]
                self.forbidden_pairs.update(forbidden_pairs)
        
        # Mark remaining pairs as forbidden (only if confident - 3+ correct words)
        if correct_words >= 3:
            remaining_pairs = [pair for pair in all_pairs if pair not in protected_pairs]
            self.forbidden_pairs.update(remaining_pairs)
    
    def _find_best_trio(self, group_upper: List[str]) -> Optional[tuple]:
        """
        Find the 3 words that form the most cohesive group.
        
        Args:
            group_upper: Uppercased words from the group
            
        Returns:
            Best trio (tuple of 3 words) or None
        """
        best_trio = None
        best_score = float('-inf')
        
        for trio in combinations(group_upper, 3):
            # Average similarity within trio
            trio_score = sum(
                self.similarity_fn.similarity(w1, w2)
                for w1, w2 in combinations(trio, 2)
            ) / 3.0  # C(3,2) = 3 pairs
            
            if trio_score > best_score:
                best_score = trio_score
                best_trio = trio
        
        return best_trio

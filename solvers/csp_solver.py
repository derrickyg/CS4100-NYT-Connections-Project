"""
solver using a csp approach
"""
from typing import List, Dict, Optional, Set, Tuple
from itertools import combinations
from similarity.embedding_similarity import EmbeddingSimilarity
from evaluation.game_simulator import GameSimulator, GameFeedback
import random


class CSP:
    """Represents a Constraint Satisfaction Problem."""
    
    def __init__(self, words: List[str]):
        """
        Initialize CSP for Connections puzzle.
        
        Args:
            words: List of words to assign to groups
        """
        self.variables = [w.upper() for w in words]  # Each word is a variable
        self.known_pairs: Set[Tuple[str, str]] = set()  # Words that must be together
        self.forbidden_pairs: Set[Tuple[str, str]] = set()  # Words that cannot be together
    


class CSPSolver:
    """the actual solver"""
    
    def __init__(self, similarity_function: EmbeddingSimilarity):
        """
        Initialize CSP solver.
        
        Args:
            similarity_function: Function to compute word similarity
        """
        self.similarity_fn = similarity_function
        self.csp: Optional[CSP] = None
        self.submission_history: List[Dict] = []
        # Track groups we've already tried (to avoid repeats)
        self.tried_groups: Set[tuple] = set()
    
    def solve_with_feedback(self, game: GameSimulator) -> Dict:
        """
        Solve puzzle using CSP constraint-guided search with feedback learning.
        
        Args:
            game: GameSimulator instance
            
        Returns:
            Dictionary with solution and statistics
        """
        submissions = []
        
        # Initialize CSP with remaining words
        remaining_words = game.get_remaining_words()
        if not remaining_words:
            return {
                "solved_groups": game.get_solved_groups(),
                "submissions": [],
                "total_submissions": 0,
                "mistakes": 0,
                "is_won": True
            }
        
        self.csp = CSP(remaining_words)
        self.submission_history.clear()
        self.tried_groups.clear()
        
        while not game.is_game_over:
            remaining = game.get_remaining_words()
            if len(remaining) == 0:
                break
            
            # Update CSP with current remaining words
            remaining_upper = set(w.upper() for w in remaining)
            if self.csp is None or remaining_upper != set(self.csp.variables):
                # Preserve learned constraints before creating new CSP
                old_known_pairs = self.csp.known_pairs.copy() if self.csp else set()
                old_forbidden_pairs = self.csp.forbidden_pairs.copy() if self.csp else set()
                
                # Create new CSP
                self.csp = CSP(remaining)
                
                # Preserve learned constraints (only pairs where both words are still remaining)
                self.csp.known_pairs = {
                    (w1, w2) for w1, w2 in old_known_pairs
                    if w1 in remaining_upper and w2 in remaining_upper
                }
                self.csp.forbidden_pairs = {
                    (w1, w2) for w1, w2 in old_forbidden_pairs
                    if w1 in remaining_upper and w2 in remaining_upper
                }
            
            # Strategy 1: Handle partial matches (3/4 or 2/4 correct) - CRITICAL for success!
            group = self._handle_partial_matches(remaining)
            
            # Strategy 2: Find next group using CSP constraints
            if not group:
                group = self._find_next_group_incremental(remaining)
            
            # Strategy 3: Fallback to similarity-based selection
            if not group or len(group) != 4:
                group = self._get_best_guess_constrained(remaining)
            
            # Ensure we haven't tried this group before
            group_tuple = self._normalize_group(group)
            max_attempts = 50
            attempts = 0
            while group_tuple in self.tried_groups and attempts < max_attempts:
                group = self._get_best_guess_constrained(remaining)
                if group:
                    group_tuple = self._normalize_group(group)
                attempts += 1
            
            if group:
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
            self._learn_from_feedback(group, feedback)
            
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
    
    def _normalize_group(self, group: List[str]) -> tuple:
        """Normalize group for comparison."""
        return tuple(sorted(w.upper() for w in group))
    
    def _handle_partial_matches(self, remaining: List[str]) -> Optional[List[str]]:
        """
        Handle partial matches (3/4 or 2/4 correct) - this is critical for success!
        Similar to iterative solver's approach.
        
        Args:
            remaining: Remaining words
            
        Returns:
            Refined group or None
        """
        if len(remaining) < 4:
            return None
        
        # Check recent submissions for partial matches
        for correct_count in [3, 2]:
            history_size = 10 if correct_count == 3 else 5
            for submission in reversed(self.submission_history[-history_size:]):
                if (not submission['feedback'].is_correct and 
                    submission['feedback'].correct_words == correct_count):
                    if correct_count == 3:
                        group = self._refine_partial_match(submission['group'], remaining)
                    else:
                        group = self._refine_two_correct(submission['group'], remaining)
                    
                    if group and self._normalize_group(group) not in self.tried_groups:
                        return group
        
        return None
    
    def _refine_partial_match(self, previous_group: List[str], remaining_words: List[str]) -> Optional[List[str]]:
        """Refine a group that had 3/4 correct by swapping the wrong word."""
        remaining_set = set(w.upper() for w in remaining_words)
        available_from_prev = [w for w in previous_group if w.upper() in remaining_set]
        
        if len(available_from_prev) < 3:
            return None
        
        # Try all combinations of 3 words from previous group
        best_group = None
        best_score = float('-inf')
        
        for three_words in combinations(available_from_prev, 3):
            three_words_set = set(w.upper() for w in three_words)
            candidates = [w for w in remaining_words if w.upper() not in three_words_set]
            
            # Find best completion for this combination
            group = self._find_best_completion_constrained(list(three_words), candidates)
            if group and self._normalize_group(group) not in self.tried_groups:
                # Score this group
                score = sum(
                    self.similarity_fn.similarity(w1, w2)
                    for w1, w2 in combinations(group, 2)
                )
                if score > best_score:
                    best_score = score
                    best_group = group
        
        return best_group
    
    def _refine_two_correct(self, previous_group: List[str], remaining_words: List[str]) -> Optional[List[str]]:
        """When we got 2/4 correct, try different combinations of those 2 words."""
        remaining_set = set(w.upper() for w in remaining_words)
        available_from_prev = [w for w in previous_group if w.upper() in remaining_set]
        
        if len(available_from_prev) < 2:
            return None
        
        # Try all pairs from the previous group
        for pair in combinations(available_from_prev, 2):
            pair_set = set(w.upper() for w in pair)
            candidates = [w for w in remaining_words if w.upper() not in pair_set]
            
            if len(candidates) < 2:
                continue
            
            # Find best combination of 2 more words
            best_group = self._find_best_completion_constrained(list(pair), candidates)
            if best_group and self._normalize_group(best_group) not in self.tried_groups:
                return best_group
        
        return None
    
    def _find_next_group_incremental(self, remaining: List[str]) -> Optional[List[str]]:
        """
        Find next group incrementally using CSP constraints.
        This is much faster than solving all words at once.
        
        Args:
            remaining: Remaining words to choose from
            
        Returns:
            Group of 4 words or None
        """
        if len(remaining) < 4:
            return remaining if remaining else None
        
        # Strategy 1: If we have known pairs, try to build a group from them
        if self.csp.known_pairs:
            group = self._build_group_from_known_pairs(remaining)
            if group and self._normalize_group(group) not in self.tried_groups:
                return group
        
        # Strategy 2: Fall back to similarity-based selection
        return None
    
    def _build_group_from_known_pairs(self, remaining: List[str]) -> Optional[List[str]]:
        """Build a group from known pairs."""
        remaining_set = set(w.upper() for w in remaining)
        
        # Find a known pair where both words are still available
        for w1, w2 in self.csp.known_pairs:
            if w1 in remaining_set and w2 in remaining_set:
                # Try to find 2 more words that work with this pair
                base = [w1, w2]
                candidates = [w for w in remaining if w.upper() not in {w1, w2}]
                
                # Find best completion using similarity and constraints
                best_completion = self._find_best_completion_constrained(base, candidates)
                if best_completion:
                    return best_completion
        
        return None
    
    def _find_best_completion_constrained(self, base: List[str], candidates: List[str]) -> Optional[List[str]]:
        """Find best completion for a base group using constraints and similarity."""
        if len(base) >= 4:
            return base[:4]
        
        needed = 4 - len(base)
        if len(candidates) < needed:
            return None
        
        base_set = set(w.upper() for w in base)
        
        # Score candidates by similarity and constraint satisfaction
        scored = []
        for candidate in candidates:
            # Check if candidate violates any forbidden pairs
            violates = any(
                (candidate.upper() == w1 and w2 in base_set) or 
                (candidate.upper() == w2 and w1 in base_set)
                for w1, w2 in self.csp.forbidden_pairs
            )
            if violates:
                continue
            
            # Score by similarity to base words
            avg_sim = sum(
                self.similarity_fn.similarity(candidate, base_word)
                for base_word in base
            ) / len(base) if base else 0.0
            
            # Bonus for known pairs
            pair_bonus = sum(
                1.0 for w1, w2 in self.csp.known_pairs
                if (candidate.upper() == w1 and w2 in base_set) or
                   (candidate.upper() == w2 and w1 in base_set)
            )
            
            scored.append((candidate, avg_sim + pair_bonus))
        
        if len(scored) < needed:
            return None
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return base + [w for w, _ in scored[:needed]]
    
    def _get_best_guess_constrained(self, remaining: List[str]) -> List[str]:
        """Get best guess using similarity, filtered by CSP constraints."""
        if len(remaining) < 4:
            return remaining
        
        # Find group with highest pairwise similarity that satisfies constraints
        best_group = None
        best_score = float('-inf')
        
        from itertools import combinations
        # Use prioritized combinations like iterative solver
        all_combos = list(combinations(remaining, 4))
        max_combos = min(200, len(all_combos))
        
        # Prioritize combos with known pairs
        combos_with_pairs = []
        combos_without_pairs = []
        
        for combo in all_combos:
            combo_set = set(w.upper() for w in combo)
            has_pair = any(
                w1 in combo_set and w2 in combo_set
                for w1, w2 in self.csp.known_pairs
            )
            
            if has_pair:
                combos_with_pairs.append(combo)
            else:
                combos_without_pairs.append(combo)
        
        # Sample more from combos with known pairs
        half_max = max_combos // 2
        prioritized = random.sample(combos_with_pairs, min(half_max, len(combos_with_pairs)))
        
        # Fill remaining slots from combos without pairs
        remaining_slots = max_combos - len(prioritized)
        if remaining_slots > 0 and len(combos_without_pairs) > 0:
            sample_size = min(remaining_slots, len(combos_without_pairs))
            prioritized.extend(random.sample(combos_without_pairs, sample_size))
        
        checked = 0
        for combo in prioritized:
            checked += 1
            if checked > max_combos:
                break
            
            # Skip if already tried
            if self._normalize_group(list(combo)) in self.tried_groups:
                continue
            
            # Check constraints
            if self._violates_constraints(combo):
                continue
            
            score = sum(
                self.similarity_fn.similarity(w1, w2)
                for w1, w2 in combinations(combo, 2)
            )
            
            # Bonus for known pairs
            combo_set = set(w.upper() for w in combo)
            pair_bonus = sum(
                2.0 for w1, w2 in self.csp.known_pairs
                if w1 in combo_set and w2 in combo_set
            )
            
            score += pair_bonus
            
            if score > best_score:
                best_score = score
                best_group = list(combo)
        
        # Fallback if no constrained group found
        if best_group is None:
            return self._get_best_guess(remaining)
        
        return best_group
    
    def _violates_constraints(self, combo: tuple) -> bool:
        """Check if a combination violates CSP constraints."""
        combo_set = set(w.upper() for w in combo)
        
        # Check forbidden pairs
        for w1, w2 in self.csp.forbidden_pairs:
            if w1 in combo_set and w2 in combo_set:
                return True
        
        return False
    
    def _get_best_guess(self, remaining: List[str]) -> List[str]:
        """Get best guess using similarity when CSP fails."""
        if len(remaining) < 4:
            return remaining
        
        # Find group with highest pairwise similarity
        best_group = None
        best_score = float('-inf')
        
        from itertools import combinations
        # Limit to reasonable number of combinations
        max_combos = 100
        checked = 0
        
        for combo in combinations(remaining, 4):
            checked += 1
            if checked > max_combos:
                break
            
            score = sum(
                self.similarity_fn.similarity(w1, w2)
                for w1, w2 in combinations(combo, 2)
            )
            if score > best_score:
                best_score = score
                best_group = list(combo)
        
        if best_group is None:
            # Last resort: return first 4 words
            return remaining[:4]
        
        return best_group
    
    def _learn_from_feedback(self, group: List[str], feedback: GameFeedback):
        """
        Learn constraints from feedback.
        
        Args:
            group: Submitted group
            feedback: Feedback received
        """
        if self.csp is None:
            return
        
        group_upper = [w.upper() for w in group]
        all_pairs = list(combinations(group_upper, 2))
        
        if feedback.is_correct:
            # All words in this group belong together
            self.csp.known_pairs.update(all_pairs)
        elif feedback.correct_words == 0:
            # None of these words belong together
            self.csp.forbidden_pairs.update(all_pairs)
        elif feedback.correct_words >= 2:
            # Some words belong together
            self._learn_from_partial_feedback(group_upper, all_pairs, feedback.correct_words)
    
    def _learn_from_partial_feedback(self, group_upper: List[str], all_pairs: List[Tuple[str, str]], correct_words: int):
        """Learn from partial feedback (2/4 or 3/4 correct)."""
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
        self.csp.known_pairs.update(protected_pairs)
        
        # For 3/4 matches, find the best trio (compute once, reuse)
        if correct_words >= 3:
            best_trio = self._find_best_trio(group_upper)
            if best_trio:
                trio_pairs = list(combinations(best_trio, 2))
                self.csp.known_pairs.update(trio_pairs)
                
                # Mark pairs involving the 4th word as forbidden
                bottom_word = [w for w in group_upper if w not in best_trio][0]
                forbidden_pairs = [(bottom_word, w) for w in best_trio]
                self.csp.forbidden_pairs.update(forbidden_pairs)
    
    def _find_best_trio(self, group_upper: List[str]) -> Optional[Tuple[str, str, str]]:
        """Find the 3 words that form the most cohesive group."""
        best_trio = None
        best_score = float('-inf')
        
        for trio in combinations(group_upper, 3):
            trio_score = sum(
                self.similarity_fn.similarity(w1, w2)
                for w1, w2 in combinations(trio, 2)
            ) / 3.0  # C(3,2) = 3 pairs
            
            if trio_score > best_score:
                best_score = trio_score
                best_trio = trio
        
        return best_trio


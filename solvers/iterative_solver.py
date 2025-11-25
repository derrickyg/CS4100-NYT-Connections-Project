"""
Iterative solver that uses game feedback to improve guesses.
"""
from typing import List, Dict, Optional, Set
from itertools import combinations
from similarity.combined_similarity import CombinedSimilarity
from evaluation.game_simulator import GameSimulator, GameFeedback
import random
import config


class IterativeSolver:
    """Solver that submits groups iteratively using game feedback."""
    
    def __init__(self, similarity_function: CombinedSimilarity):
        """
        Initialize iterative solver.
        
        Args:
            similarity_function: Function to compute word similarity
        """
        self.similarity_fn = similarity_function
        # Track which words we know belong together (from feedback)
        self.known_pairs: Set[tuple] = set()
        # Track which words we know don't belong together
        self.forbidden_pairs: Set[tuple] = set()
        # Track submission history for learning
        self.submission_history: List[Dict] = []
        # Track groups we've already tried (to avoid repeats)
        self.tried_groups: Set[tuple] = set()
    
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
            # Get remaining words
            remaining = game.get_remaining_words()
            
            if len(remaining) == 0:
                break
            
            # Select next group to submit (avoid repeats)
            max_attempts = 100  # Prevent infinite loop
            attempt = 0
            group = None
            
            while attempt < max_attempts:
                candidate = self._select_next_group(remaining, game, self.tried_groups)
                # Normalize group to tuple for comparison
                candidate_tuple = tuple(sorted(w.upper() for w in candidate))
                
                if candidate_tuple not in self.tried_groups:
                    group = candidate
                    self.tried_groups.add(candidate_tuple)
                    break
                attempt += 1
            
            # Fallback if all groups tried - try random untried combinations
            if group is None:
                remaining_list = list(remaining)
                if len(remaining_list) >= 4:
                    # Try random combinations until we find one we haven't tried
                    for _ in range(50):
                        candidate = random.sample(remaining_list, 4)
                        candidate_tuple = tuple(sorted(w.upper() for w in candidate))
                        if candidate_tuple not in self.tried_groups:
                            group = candidate
                            self.tried_groups.add(candidate_tuple)
                            break
                    
                    # Last resort: just pick first 4
                    if group is None:
                        group = remaining_list[:4]
                else:
                    group = remaining_list
            
            # Ensure we have exactly 4 words (or fewer if not enough remaining)
            if len(group) > 4:
                group = group[:4]
            elif len(group) < 4 and len(remaining) >= 4:
                # This shouldn't happen, but add safety check
                remaining_list = list(remaining)
                group = remaining_list[:4] if len(remaining_list) >= 4 else remaining_list
            
            # Submit group
            print(f"  Submission {len(submissions) + 1}: {', '.join(group)}")
            feedback = game.submit_group(group)
            submission_data = {
                "group": group,
                "feedback": feedback
            }
            submissions.append(submission_data)
            self.submission_history.append(submission_data)
            
            # Learn from feedback
            self._learn_from_feedback(group, feedback, game)
            
            # Print feedback
            if feedback.is_correct:
                print(f"    → ✓ Correct! Group {feedback.group_id} solved")
            else:
                print(f"    → ✗ Incorrect ({feedback.correct_words}/4 words correct)")
        
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
        
        # Strategy 1: If we got 3/4 correct recently, try variations of that group
        # Check last few submissions for 3/4 matches - prioritize most recent
        for submission in reversed(self.submission_history[-10:]):  # Check last 10 submissions
            if (not submission['feedback'].is_correct and 
                submission['feedback'].correct_words == 3):
                # Try swapping the incorrect word - try multiple candidates
                group = self._refine_partial_match(submission['group'], remaining_words, tried_groups)
                if group:
                    # Verify we haven't tried this exact group
                    group_tuple = tuple(sorted(w.upper() for w in group))
                    if group_tuple not in tried_groups:
                        return group
        
        # Strategy 1b: If we got 2/4 correct, try variations with different combinations
        for submission in reversed(self.submission_history[-5:]):  # Check more submissions
            if (not submission['feedback'].is_correct and 
                submission['feedback'].correct_words == 2):
                # Try building groups from pairs in this submission
                group = self._refine_two_correct(submission['group'], remaining_words, tried_groups)
                if group:
                    group_tuple = tuple(sorted(w.upper() for w in group))
                    if group_tuple not in tried_groups:
                        return group
        
        # Strategy 2: If we have known pairs, try to complete them
        if self.known_pairs:
            group = self._build_group_from_pairs(remaining_words, tried_groups)
            if group:
                return group
        
        # Strategy 3: Use similarity to find most cohesive group
        group = self._find_most_similar_group(remaining_words, tried_groups)
        
        # Verify the group hasn't been tried (should be handled by _find_most_similar_group, but double-check)
        if group:
            group_tuple = tuple(sorted(w.upper() for w in group))
            if group_tuple not in tried_groups:
                return group
        
        # If all strategies failed, use random untried combinations (fast fallback)
        if len(remaining_words) >= 4:
            remaining_list = list(remaining_words)
            for _ in range(20):  # Try up to 20 random combinations
                candidate = random.sample(remaining_list, 4)
                candidate_tuple = tuple(sorted(w.upper() for w in candidate))
                if candidate_tuple not in tried_groups:
                    return candidate
        
        # If all else fails, return first 4 words (will be marked as tried)
        return remaining_words[:4] if len(remaining_words) >= 4 else remaining_words
    
    def _refine_partial_match(self, previous_group: List[str], remaining_words: List[str], 
                            tried_groups: Set[tuple]) -> Optional[List[str]]:
        """
        Try to refine a group that had 3/4 correct by swapping the wrong word.
        Tries multiple variations to find the correct 4th word.
        
        Args:
            previous_group: Previous submission that had 3/4 correct
            remaining_words: Words still available
            
        Returns:
            Refined group of exactly 4 words or None
        """
        # Find which words from previous group are still available
        remaining_set = set(w.upper() for w in remaining_words)
        available_from_prev = [w for w in previous_group if w.upper() in remaining_set]
        
        # We need exactly 3 words from the previous group (the 3 that were correct)
        # Since we don't know which 3 were correct, try all combinations of 3
        if len(available_from_prev) < 3:
            return None
        
        # If we have exactly 3 available, use those
        # If we have 4 available (all still unsolved), we need to pick 3
        if len(available_from_prev) == 3:
            # Perfect - we have the 3 correct words
            used_words = set(w.upper() for w in available_from_prev)
            candidates = [w for w in remaining_words if w.upper() not in used_words]
        else:
            # We have 4 words available, but only 3 were correct
            # Try all combinations of 3 from the previous group
            best_group = None
            best_score = float('-inf')
            
            for three_words in combinations(available_from_prev, 3):
                used_words = set(w.upper() for w in three_words)
                candidates = [w for w in remaining_words if w.upper() not in used_words]
                
                if not candidates:
                    continue
                
                # Score candidates
                scores = []
                for candidate in candidates:
                    test_group = list(three_words) + [candidate]
                    test_tuple = tuple(sorted(w.upper() for w in test_group))
                    if test_tuple in tried_groups:
                        continue
                    
                    avg_sim = sum(
                        self.similarity_fn.similarity(candidate, word) 
                        for word in three_words
                    ) / 3
                    
                    # Bonus if candidate forms known pairs with any of the 3 words
                    pair_bonus = 0.0
                    for word in three_words:
                        pair = tuple(sorted([candidate.upper(), word.upper()]))
                        if pair in self.known_pairs:
                            pair_bonus += 1.0
                    
                    scores.append((candidate, avg_sim + pair_bonus))
                
                if scores:
                    scores.sort(key=lambda x: x[1], reverse=True)
                    # Try top 3 candidates for this combination
                    for candidate, score in scores[:3]:
                        refined_group = list(three_words) + [candidate]
                        refined_tuple = tuple(sorted(w.upper() for w in refined_group))
                        if refined_tuple not in tried_groups:
                            if score > best_score:
                                best_score = score
                                best_group = refined_group
                                break  # Found a good candidate for this combination
            
            if best_group:
                return best_group
            return None
        
        if not candidates:
            return None
        
        # Try multiple candidates, sorted by similarity
        # Try ALL candidates, not just the first one that works
        scores = []
        for candidate in candidates:
            # Check if this combination was already tried
            test_group = available_from_prev + [candidate]
            if len(test_group) != 4:
                continue  # Safety check
            test_tuple = tuple(sorted(w.upper() for w in test_group))
            if test_tuple in tried_groups:
                continue  # Skip if already tried
            
            # Score by average similarity to the 3 known words
            avg_sim = sum(
                self.similarity_fn.similarity(candidate, word) 
                for word in available_from_prev
            ) / len(available_from_prev)
            
            # Also check if candidate forms known pairs with any of the 3 words
            pair_bonus = 0.0
            for word in available_from_prev:
                pair = tuple(sorted([candidate.upper(), word.upper()]))
                if pair in self.known_pairs:
                    pair_bonus += 1.0
            
            scores.append((candidate, avg_sim + pair_bonus))
        
        if not scores:
            return None
        
        # Sort by similarity and try the best ones (try top 5 candidates)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best candidate that hasn't been tried
        for candidate, score in scores[:5]:  # Try top 5 candidates
            refined_group = available_from_prev + [candidate]
            if len(refined_group) != 4:
                continue  # Safety check
            refined_tuple = tuple(sorted(w.upper() for w in refined_group))
            if refined_tuple not in tried_groups:
                return refined_group
        
        return None
    
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
        remaining_set = set(w.upper() for w in remaining_words)
        available_from_prev = [w for w in previous_group if w.upper() in remaining_set]
        
        if len(available_from_prev) < 2:
            return None
        
        # Try all pairs from the previous group
        for pair in combinations(available_from_prev, 2):
            used_words = set(w.upper() for w in pair)
            candidates = [w for w in remaining_words if w.upper() not in used_words]
            
            if len(candidates) < 2:
                continue
            
            # Score all combinations of 2 more words
            best_group = None
            best_score = float('-inf')
            
            for two_more in combinations(candidates, 2):
                test_group = list(pair) + list(two_more)
                test_tuple = tuple(sorted(w.upper() for w in test_group))
                
                if test_tuple in tried_groups:
                    continue
                
                # Score by similarity
                score = 0.0
                for w1 in test_group:
                    for w2 in test_group:
                        if w1 != w2:
                            score += self.similarity_fn.similarity(w1, w2)
                
                if score > best_score:
                    best_score = score
                    best_group = test_group
            
            if best_group:
                return best_group
        
        return None
    
    def _build_group_from_pairs(self, remaining_words: List[str], tried_groups: Set[tuple]) -> Optional[List[str]]:
        """
        Try to build a group from known pairs.
        
        Args:
            remaining_words: Available words
            
        Returns:
            Group of 4 words or None
        """
        remaining_set = set(w.upper() for w in remaining_words)
        
        # Find pairs where both words are still available
        available_pairs = [
            (w1, w2) for w1, w2 in self.known_pairs
            if w1.upper() in remaining_set and w2.upper() in remaining_set
        ]
        
        if not available_pairs:
            return None
        
        # Try to combine pairs into a group
        # Simple approach: take first pair and find 2 more similar words
        if available_pairs:
            w1, w2 = available_pairs[0]
            group = [w1, w2]
            
            # Find 2 more words similar to this pair
            remaining = [w for w in remaining_words if w.upper() not in {w1.upper(), w2.upper()}]
            
            if len(remaining) >= 2:
                # Score remaining words by similarity to the pair
                scores = []
                for word in remaining:
                    sim1 = self.similarity_fn.similarity(word, w1)
                    sim2 = self.similarity_fn.similarity(word, w2)
                    scores.append((word, (sim1 + sim2) / 2))
                
                scores.sort(key=lambda x: x[1], reverse=True)
                group.extend([scores[0][0], scores[1][0]])
                
                # Check if we've tried this group
                group_tuple = tuple(sorted(w.upper() for w in group))
                if group_tuple not in tried_groups:
                    return group
        
        return None
    
    def _find_most_similar_group(self, remaining_words: List[str], tried_groups: Set[tuple]) -> List[str]:
        """
        Find the group of 4 words with highest similarity.
        
        Args:
            remaining_words: Available words
            
        Returns:
            Group of 4 words
        """
        if len(remaining_words) == 4:
            return remaining_words
        
        # Try all combinations (limit to avoid timeout)
        max_combinations = config.ITERATIVE_MAX_COMBINATIONS
        all_combos = list(combinations(remaining_words, 4))
        
        if len(all_combos) > max_combinations:
            # Prioritize combinations with known pairs
            combos_with_pairs = []
            combos_without_pairs = []
            
            for combo in all_combos:
                combo_set = set(w.upper() for w in combo)
                has_known_pair = any(
                    w1.upper() in combo_set and w2.upper() in combo_set
                    for w1, w2 in self.known_pairs
                )
                
                if has_known_pair:
                    combos_with_pairs.append(combo)
                else:
                    combos_without_pairs.append(combo)
            
            # Sample more from combos with known pairs
            if len(combos_with_pairs) > max_combinations // 2:
                all_combos = random.sample(combos_with_pairs, max_combinations // 2)
                all_combos.extend(random.sample(combos_without_pairs, max_combinations // 2))
            else:
                all_combos = combos_with_pairs
                all_combos.extend(random.sample(combos_without_pairs, 
                                                max_combinations - len(combos_with_pairs)))
        
        best_group = None
        best_score = float('-inf')
        
        # Limit how many combinations we check to avoid slowdown
        checked = 0
        max_to_check = min(len(all_combos), config.ITERATIVE_MAX_TO_CHECK)
        
        for combo in all_combos:
            if checked >= max_to_check:
                break
            checked += 1
            
            # Check if already tried
            combo_tuple = tuple(sorted(w.upper() for w in combo))
            if combo_tuple in tried_groups:
                continue
            
            # Check if this combo violates any forbidden pairs
            combo_set = set(w.upper() for w in combo)
            violates = False
            for w1, w2 in self.forbidden_pairs:
                if w1.upper() in combo_set and w2.upper() in combo_set:
                    violates = True
                    break
            
            if violates:
                continue
            
            # Compute within-group similarity
            score = 0.0
            for i, word1 in enumerate(combo):
                for word2 in combo[i+1:]:
                    score += self.similarity_fn.similarity(word1, word2)
            
            # Bonus for known pairs
            for w1, w2 in self.known_pairs:
                if w1.upper() in combo_set and w2.upper() in combo_set:
                    score += config.ITERATIVE_KNOWN_PAIR_BONUS
            
            if score > best_score:
                best_score = score
                best_group = list(combo)
        
        # If we found a good group, return it
        if best_group:
            return best_group
        
        # If no good group found, return a random untried combination
        remaining_list = list(remaining_words)
        for _ in range(20):  # Try up to 20 random combinations
            candidate = random.sample(remaining_list, 4)
            candidate_tuple = tuple(sorted(w.upper() for w in candidate))
            if candidate_tuple not in tried_groups:
                return candidate
        
        # Last resort: return first 4 words
        return remaining_words[:4]
    
    def _learn_from_feedback(self, group: List[str], feedback: GameFeedback, game: GameSimulator):
        """
        Learn from feedback to improve future guesses.
        
        Args:
            group: Submitted group
            feedback: Feedback received
            game: Game simulator
        """
        if feedback.is_correct:
            # All words in this group belong together
            group_upper = [w.upper() for w in group]
            for i, w1 in enumerate(group_upper):
                for w2 in group_upper[i+1:]:
                    self.known_pairs.add((w1, w2))
        else:
            # Partial feedback: some words belong together, some don't
            group_upper = [w.upper() for w in group]
            
            if feedback.correct_words == 0:
                # None of these words belong together
                for i, w1 in enumerate(group_upper):
                    for w2 in group_upper[i+1:]:
                        self.forbidden_pairs.add((w1, w2))
            elif feedback.correct_words >= 2:
                # At least 2 words belong together - identify likely pairs
                # Find the most similar pairs (likely to be the correct ones)
                pair_scores = []
                for i, w1 in enumerate(group_upper):
                    for w2 in group_upper[i+1:]:
                        sim = self.similarity_fn.similarity(w1, w2)
                        pair_scores.append(((w1, w2), sim))
                
                # Sort by similarity and take top pairs
                pair_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Number of pairs we expect: C(correct_words, 2)
                num_correct_pairs = feedback.correct_words * (feedback.correct_words - 1) // 2
                num_pairs_to_protect = min(num_correct_pairs, len(pair_scores))
                
                # Add top pairs as known (strong signal) - be more aggressive
                for (w1, w2), _ in pair_scores[:num_pairs_to_protect]:
                    self.known_pairs.add((w1, w2))
                
                # For 3/4 matches, be more aggressive: assume the 3 most similar words are correct
                if feedback.correct_words == 3:
                    # Find the 3 words that form the most cohesive group
                    # Try all combinations of 3 and find the one with highest average similarity
                    best_trio = None
                    best_trio_score = float('-inf')
                    
                    for trio in combinations(group_upper, 3):
                        # Compute average similarity within trio
                        trio_sim = sum(
                            self.similarity_fn.similarity(w1, w2)
                            for w1, w2 in combinations(trio, 2)
                        ) / 3.0  # C(3,2) = 3 pairs
                        
                        if trio_sim > best_trio_score:
                            best_trio_score = trio_sim
                            best_trio = trio
                    
                    if best_trio:
                        # Add all pairs among the best trio as known
                        best_trio_list = list(best_trio)
                        for i, w1 in enumerate(best_trio_list):
                            for w2 in best_trio_list[i+1:]:
                                self.known_pairs.add((w1, w2))
                        
                        # Mark pairs involving the 4th word as forbidden
                        bottom_word = [w for w in group_upper if w not in best_trio][0]
                        for w in best_trio:
                            self.forbidden_pairs.add((bottom_word, w))
                
                # Mark remaining pairs as forbidden (weaker signal - might be wrong)
                protected_pairs = {pair for pair, _ in pair_scores[:num_pairs_to_protect]}
                for i, w1 in enumerate(group_upper):
                    for w2 in group_upper[i+1:]:
                        if (w1, w2) not in protected_pairs and (w2, w1) not in protected_pairs:
                            # Only add as forbidden if we're confident (3+ correct words)
                            if feedback.correct_words >= 3:
                                self.forbidden_pairs.add((w1, w2))


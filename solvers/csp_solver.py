"""
CSP Solver with backtracking for NYT Connections puzzles.
"""
from typing import List, Dict, Set, Optional
from similarity.combined_similarity import CombinedSimilarity
import config


class CSPSolver:
    """CSP solver using backtracking with constraint propagation."""
    
    def __init__(self, similarity_function: CombinedSimilarity):
        """
        Initialize CSP solver.
        
        Args:
            similarity_function: Function to compute word similarity
        """
        self.similarity_fn = similarity_function
    
    def solve(self, words: List[str]) -> Optional[Dict[int, List[str]]]:
        """
        Solve a Connections puzzle using CSP backtracking.
        
        Args:
            words: List of 16 words
            
        Returns:
            Dictionary mapping group_id to list of words, or None if no solution
        """
        if len(words) != 16:
            raise ValueError(f"Expected 16 words, got {len(words)}")
        
        # Initialize groups
        groups = {1: [], 2: [], 3: [], 4: []}
        unassigned_words = set(words)
        
        # Try to find solution
        solution = self._backtrack(groups, unassigned_words)
        
        return solution
    
    def _backtrack(self, groups: Dict[int, List[str]], 
                   unassigned_words: Set[str]) -> Optional[Dict[int, List[str]]]:
        """
        Backtracking search with constraint propagation.
        
        Args:
            groups: Current group assignments
            unassigned_words: Words not yet assigned
            
        Returns:
            Solution dictionary or None
        """
        # Base case: all words assigned
        if len(unassigned_words) == 0:
            if self._validate_solution(groups):
                return groups.copy()
            return None
        
        # Select next word using MRV heuristic
        word = self._select_unassigned_word(unassigned_words, groups)
        
        # Order domain values (groups) by least constraining value
        ordered_groups = self._order_domain_values(word, groups)
        
        for group_id in ordered_groups:
            # Check if assignment is consistent
            if self._is_consistent(word, group_id, groups):
                # Make assignment
                groups[group_id].append(word)
                unassigned_words.remove(word)
                
                # Forward checking: if group is full, propagate constraints
                if len(groups[group_id]) == 4:
                    self._propagate_constraints(groups, unassigned_words)
                
                # Recurse
                result = self._backtrack(groups, unassigned_words)
                if result is not None:
                    return result
                
                # Backtrack
                groups[group_id].remove(word)
                unassigned_words.add(word)
        
        return None
    
    def _select_unassigned_word(self, unassigned_words: Set[str], 
                                groups: Dict[int, List[str]]) -> str:
        """
        Select next word using MRV (Minimum Remaining Values) heuristic.
        Counts how many groups each word can be consistently assigned to.
        
        Args:
            unassigned_words: Set of unassigned words
            groups: Current group assignments
            
        Returns:
            Selected word
        """
        # Count valid groups for each word based on consistency
        min_valid_groups = float('inf')
        best_word = None
        
        for word in unassigned_words:
            # Count groups that are not full AND where assignment would be consistent
            valid_groups = 0
            for group_id in [1, 2, 3, 4]:
                if len(groups[group_id]) < 4:
                    if self._is_consistent(word, group_id, groups):
                        valid_groups += 1
            
            if valid_groups < min_valid_groups:
                min_valid_groups = valid_groups
                best_word = word
        
        # If all words have same number of valid groups, pick one with highest similarity to existing groups
        if best_word is None or min_valid_groups == float('inf'):
            # Fallback: pick word with highest average similarity to existing groups
            best_word = None
            best_score = float('-inf')
            for word in unassigned_words:
                score = 0.0
                count = 0
                for group_id in [1, 2, 3, 4]:
                    if len(groups[group_id]) > 0:
                        score += self.similarity_fn.average_similarity(word, groups[group_id])
                        count += 1
                if count > 0:
                    score /= count
                if score > best_score:
                    best_score = score
                    best_word = word
        
        return best_word if best_word else list(unassigned_words)[0]
    
    def _order_domain_values(self, word: str, groups: Dict[int, List[str]]) -> List[int]:
        """
        Order group assignments by least constraining value (best fit first).
        
        Args:
            word: Word to assign
            groups: Current group assignments
            
        Returns:
            Ordered list of group IDs
        """
        group_scores = []
        
        for group_id in [1, 2, 3, 4]:
            if len(groups[group_id]) >= 4:
                continue
            
            # Compute average similarity to words already in group
            if len(groups[group_id]) > 0:
                avg_sim = self.similarity_fn.average_similarity(word, groups[group_id])
            else:
                avg_sim = 0.0
            
            group_scores.append((group_id, avg_sim))
        
        # Sort by similarity (descending)
        group_scores.sort(key=lambda x: x[1], reverse=True)
        return [g[0] for g in group_scores]
    
    def _is_consistent(self, word: str, group_id: int, 
                       groups: Dict[int, List[str]]) -> bool:
        """
        Check if assigning word to group maintains consistency.
        
        Args:
            word: Word to assign
            group_id: Target group ID
            groups: Current group assignments
            
        Returns:
            True if assignment is consistent
        """
        if len(groups[group_id]) >= 4:
            return False
        
        if len(groups[group_id]) == 0:
            return True
        
        # Compute average similarity to existing group members
        avg_sim = self.similarity_fn.average_similarity(word, groups[group_id])
        
        # Threshold-based consistency check
        # Use a more lenient threshold - if group is almost full, be more lenient
        threshold = config.CONSISTENCY_THRESHOLD
        if len(groups[group_id]) >= 3:
            # If group is almost full, lower threshold to allow completion
            threshold = threshold * 0.7
        
        return avg_sim > threshold
    
    def _propagate_constraints(self, groups: Dict[int, List[str]], 
                              unassigned_words: Set[str]):
        """
        Propagate constraints when a group becomes full.
        This is a simple implementation - could be enhanced with AC-3.
        
        Args:
            groups: Current group assignments
            unassigned_words: Unassigned words
        """
        # In a full implementation, we would remove the full group from
        # the domains of unassigned words. For now, this is a placeholder.
        pass
    
    def _validate_solution(self, groups: Dict[int, List[str]]) -> bool:
        """
        Validate that solution satisfies all constraints.
        
        Args:
            groups: Group assignments
            
        Returns:
            True if valid solution
        """
        # Check that each group has exactly 4 words
        for group_id in [1, 2, 3, 4]:
            if len(groups[group_id]) != 4:
                return False
        
        # Check that all words are assigned exactly once
        all_words = []
        for group_words in groups.values():
            all_words.extend(group_words)
        
        return len(set(all_words)) == 16 and len(all_words) == 16


"""
Hill Climbing solver with random restarts for NYT Connections puzzles.
"""
from typing import List, Dict
from itertools import combinations
import random
import copy
from similarity.combined_similarity import CombinedSimilarity
import config


class HillClimbingSolver:
    """Hill climbing solver with random restarts."""
    
    def __init__(self, similarity_function: CombinedSimilarity):
        """
        Initialize hill climbing solver.
        
        Args:
            similarity_function: Function to compute word similarity
        """
        self.similarity_fn = similarity_function
    
    def solve(self, words: List[str]) -> Dict[int, List[str]]:
        """
        Solve puzzle using hill climbing with random restarts.
        
        Args:
            words: List of 16 words
            
        Returns:
            Dictionary mapping group_id to list of words
        """
        if len(words) != 16:
            raise ValueError(f"Expected 16 words, got {len(words)}")
        
        best_solution = None
        best_score = float('-inf')
        
        for restart in range(config.HC_NUM_RESTARTS):
            # Generate random initial state
            current = self._random_initial_state(words)
            current_score = self._objective_function(current)
            
            improved = True
            iterations = 0
            
            while improved and iterations < config.HC_MAX_ITERATIONS:
                improved = False
                neighbors = self._generate_neighbors(current)
                
                # Evaluate all neighbors and pick the best one (not just first improvement)
                best_neighbor = None
                best_neighbor_score = current_score
                
                for neighbor in neighbors:
                    neighbor_score = self._objective_function(neighbor)
                    
                    if neighbor_score > best_neighbor_score:
                        best_neighbor = neighbor
                        best_neighbor_score = neighbor_score
                        improved = True
                
                if improved:
                    current = best_neighbor
                    current_score = best_neighbor_score
                
                iterations += 1
            
            if current_score > best_score:
                best_solution = current
                best_score = current_score
        
        return best_solution
    
    def _random_initial_state(self, words: List[str]) -> Dict[int, List[str]]:
        """
        Generate random initial state.
        
        Args:
            words: List of words
            
        Returns:
            Random grouping
        """
        shuffled = words.copy()
        random.shuffle(shuffled)
        
        return {
            1: shuffled[0:4],
            2: shuffled[4:8],
            3: shuffled[8:12],
            4: shuffled[12:16]
        }
    
    def _generate_neighbors(self, state: Dict[int, List[str]]) -> List[Dict[int, List[str]]]:
        """
        Generate neighbors by swapping words between groups.
        
        Args:
            state: Current state
            
        Returns:
            List of neighbor states
        """
        neighbors = []
        group_ids = [1, 2, 3, 4]
        
        # Generate neighbors by swapping words between pairs of groups
        for group1_id, group2_id in combinations(group_ids, 2):
            for word1 in state[group1_id]:
                for word2 in state[group2_id]:
                    # Create neighbor by swapping
                    neighbor = copy.deepcopy(state)
                    neighbor[group1_id].remove(word1)
                    neighbor[group1_id].append(word2)
                    neighbor[group2_id].remove(word2)
                    neighbor[group2_id].append(word1)
                    neighbors.append(neighbor)
        
        return neighbors
    
    def _objective_function(self, state: Dict[int, List[str]]) -> float:
        """
        Compute objective function score.
        Maximize within-group similarity, minimize between-group similarity.
        
        Args:
            state: Current grouping state
            
        Returns:
            Objective score
        """
        within_group_sim = 0.0
        between_group_sim = 0.0
        
        # Compute within-group similarity
        for group_id in [1, 2, 3, 4]:
            group_words = state[group_id]
            for word1, word2 in combinations(group_words, 2):
                within_group_sim += self.similarity_fn.similarity(word1, word2)
        
        # Normalize: 4 groups * C(4,2) = 4 * 6 = 24 pairs
        within_group_sim /= 24.0
        
        # Compute between-group similarity
        for group1_id, group2_id in combinations([1, 2, 3, 4], 2):
            for word1 in state[group1_id]:
                for word2 in state[group2_id]:
                    between_group_sim += self.similarity_fn.similarity(word1, word2)
        
        # Normalize: C(4,2) = 6 group pairs * 4 * 4 = 96 pairs
        between_group_sim /= 96.0
        
        # Objective: maximize within-group, minimize between-group
        score = (config.ALPHA * within_group_sim - 
                 config.BETA * between_group_sim)
        
        return score


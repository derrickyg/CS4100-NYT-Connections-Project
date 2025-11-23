"""
Simulated Annealing solver for NYT Connections puzzles.
"""
from typing import List, Dict
import random
import math
import copy
from similarity.combined_similarity import CombinedSimilarity
from solvers.hill_climbing import HillClimbingSolver
import config


class SimulatedAnnealingSolver:
    """Simulated annealing solver."""
    
    def __init__(self, similarity_function: CombinedSimilarity):
        """
        Initialize simulated annealing solver.
        
        Args:
            similarity_function: Function to compute word similarity
        """
        self.similarity_fn = similarity_function
        self.hc_solver = HillClimbingSolver(similarity_function)
    
    def solve(self, words: List[str]) -> Dict[int, List[str]]:
        """
        Solve puzzle using simulated annealing.
        
        Args:
            words: List of 16 words
            
        Returns:
            Dictionary mapping group_id to list of words
        """
        if len(words) != 16:
            raise ValueError(f"Expected 16 words, got {len(words)}")
        
        # Initialize with random state
        current = self.hc_solver._random_initial_state(words)
        current_score = self.hc_solver._objective_function(current)
        
        best = copy.deepcopy(current)
        best_score = current_score
        
        T = config.SA_INITIAL_TEMP
        
        for iteration in range(config.SA_MAX_ITERATIONS):
            # Generate random neighbor
            neighbor = self._random_neighbor(current)
            neighbor_score = self.hc_solver._objective_function(neighbor)
            
            delta = neighbor_score - current_score
            
            # Accept if better, or with probability based on temperature
            if delta > 0 or random.random() < math.exp(delta / T):
                current = neighbor
                current_score = neighbor_score
                
                if current_score > best_score:
                    best = copy.deepcopy(current)
                    best_score = current_score
            
            # Cool down
            T = T * config.SA_COOLING_RATE
            
            # Stop if temperature is very low
            if T < config.SA_MIN_TEMP:
                break
        
        return best
    
    def _random_neighbor(self, state: Dict[int, List[str]]) -> Dict[int, List[str]]:
        """
        Generate a random neighbor by swapping words between groups.
        
        Args:
            state: Current state
            
        Returns:
            Random neighbor state
        """
        neighbor = copy.deepcopy(state)
        
        # Randomly select two different groups
        group_ids = [1, 2, 3, 4]
        group1_id, group2_id = random.sample(group_ids, 2)
        
        # Randomly select one word from each group
        if len(neighbor[group1_id]) > 0 and len(neighbor[group2_id]) > 0:
            word1 = random.choice(neighbor[group1_id])
            word2 = random.choice(neighbor[group2_id])
            
            # Swap
            neighbor[group1_id].remove(word1)
            neighbor[group1_id].append(word2)
            neighbor[group2_id].remove(word2)
            neighbor[group2_id].append(word1)
        
        return neighbor


"""
Solution validation and objective score computation.
"""
from typing import Dict, List
from similarity.combined_similarity import CombinedSimilarity
from solvers.hill_climbing import HillClimbingSolver
import config


def validate_solution(groups: Dict[int, List[str]]) -> bool:
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


def compute_objective_score(groups: Dict[int, List[str]], 
                            similarity_fn: CombinedSimilarity) -> float:
    """
    Compute objective function score for a solution.
    
    Args:
        groups: Group assignments
        similarity_fn: Similarity function
        
    Returns:
        Objective score
    """
    hc_solver = HillClimbingSolver(similarity_fn)
    return hc_solver._objective_function(groups)


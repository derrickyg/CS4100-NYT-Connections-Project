"""
Evaluation metrics for Connections solver.
"""
from typing import Dict, List


def compute_accuracy(predicted_groups: Dict[int, List[str]], 
                     ground_truth_groups: Dict[int, List[str]]) -> float:
    """
    Compute partial credit accuracy.
    
    Args:
        predicted_groups: Predicted grouping
        ground_truth_groups: True grouping
        
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    num_correct_groups = 0
    
    # Convert both to sets for comparison (normalize to uppercase)
    pred_sets = [set(w.upper() for w in words) for words in predicted_groups.values()]
    truth_sets = [set(w.upper() for w in words) for words in ground_truth_groups.values()]
    
    # Count how many predicted groups exactly match ground truth groups
    for pred_set in pred_sets:
        if pred_set in truth_sets:
            num_correct_groups += 1
    
    return num_correct_groups / 4.0


def compute_word_accuracy(predicted_groups: Dict[int, List[str]], 
                         ground_truth_groups: Dict[int, List[str]]) -> float:
    """
    Compute word-level accuracy using optimal matching.
    Finds the best one-to-one matching between predicted and truth groups,
    then counts how many words are correctly grouped.
    
    Args:
        predicted_groups: Predicted grouping
        ground_truth_groups: True grouping
        
    Returns:
        Word accuracy score (0.0 to 1.0)
    """
    # Convert to uppercase sets for comparison
    pred_sets = [set(w.upper() for w in words) for words in predicted_groups.values()]
    truth_sets = [set(w.upper() for w in words) for words in ground_truth_groups.values()]
    
    # Find optimal matching using greedy approach (assign each predicted group to best unmatched truth group)
    used_truth_indices = set()
    matches = []  # List of (pred_set, truth_set) pairs
    
    # Sort predicted sets by size to match larger groups first
    pred_sets_with_idx = [(i, pred_set) for i, pred_set in enumerate(pred_sets)]
    pred_sets_with_idx.sort(key=lambda x: len(x[1]), reverse=True)
    
    for pred_idx, pred_set in pred_sets_with_idx:
        best_match_idx = None
        best_match_score = -1
        
        for truth_idx, truth_set in enumerate(truth_sets):
            if truth_idx in used_truth_indices:
                continue
            
            # Score based on intersection size
            match_score = len(pred_set & truth_set)
            if match_score > best_match_score:
                best_match_score = match_score
                best_match_idx = truth_idx
        
        if best_match_idx is not None:
            matches.append((pred_set, truth_sets[best_match_idx]))
            used_truth_indices.add(best_match_idx)
    
    # Count correct words
    correct = 0
    total = 0
    
    for pred_set, truth_set in matches:
        for word in pred_set:
            total += 1
            if word in truth_set:
                correct += 1
    
    # Handle unmatched predicted groups
    for pred_idx, pred_set in pred_sets_with_idx:
        if pred_set not in [p for p, t in matches]:
            total += len(pred_set)
    
    return correct / total if total > 0 else 0.0


def compute_exact_match(predicted_groups: Dict[int, List[str]], 
                       ground_truth_groups: Dict[int, List[str]]) -> bool:
    """
    Check if all 4 groups match exactly.
    
    Args:
        predicted_groups: Predicted grouping
        ground_truth_groups: True grouping
        
    Returns:
        True if exact match
    """
    # Normalize to uppercase sets for comparison
    pred_sets = [set(w.upper() for w in words) for words in predicted_groups.values()]
    truth_sets = [set(w.upper() for w in words) for words in ground_truth_groups.values()]
    
    if len(pred_sets) != 4 or len(truth_sets) != 4:
        return False
    
    # Check if all predicted sets are in truth sets
    for pred_set in pred_sets:
        if pred_set not in truth_sets:
            return False
    
    return True


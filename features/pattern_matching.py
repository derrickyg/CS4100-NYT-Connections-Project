"""
Detect common Connections patterns: "___ word", "word ___", compound words, etc.
"""
from typing import List, Tuple
import re


def detect_phrase_pattern(word1: str, word2: str) -> float:
    """
    Detect patterns like "___ word" or "word ___".
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        Similarity score if pattern detected, 0 otherwise
    """
    # Check if one word is a substring of the other (indicating phrase completion)
    if word1.lower() in word2.lower() or word2.lower() in word1.lower():
        return 0.3
    
    # Check for common prefixes/suffixes that form phrases
    # This is a simple heuristic - could be improved
    return 0.0


def detect_compound_word_pattern(word1: str, word2: str) -> float:
    """
    Detect if words share a common component (compound word pattern).
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        Similarity score if pattern detected
    """
    w1_lower = word1.lower()
    w2_lower = word2.lower()
    
    # Check for shared components (simple heuristic)
    # Look for common substrings of length 3 or more
    max_common_len = 0
    for i in range(len(w1_lower) - 2):
        for j in range(len(w2_lower) - 2):
            for length in range(3, min(len(w1_lower) - i, len(w2_lower) - j) + 1):
                if w1_lower[i:i+length] == w2_lower[j:j+length]:
                    max_common_len = max(max_common_len, length)
    
    if max_common_len >= 3:
        return min(max_common_len / 10.0, 0.5)
    
    return 0.0


def detect_homophone_pattern(word1: str, word2: str) -> float:
    """
    Detect if words might be homophones (sound similar).
    Simple heuristic based on similar spelling patterns.
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        Similarity score if homophone pattern detected
    """
    # Very simple heuristic: if words are similar length and have similar letters
    if abs(len(word1) - len(word2)) <= 1:
        # Count common letters
        common_letters = set(word1.lower()) & set(word2.lower())
        total_letters = set(word1.lower()) | set(word2.lower())
        
        if len(total_letters) > 0:
            similarity = len(common_letters) / len(total_letters)
            if similarity > 0.7:  # High letter overlap
                return 0.4
    
    return 0.0


def detect_category_pattern(words: List[str]) -> float:
    """
    Detect if words belong to a common category.
    This is a placeholder - could be enhanced with knowledge bases.
    
    Args:
        words: List of words to check
        
    Returns:
        Similarity score if category pattern detected
    """
    # Simple heuristic: if words are all same length or very similar
    if len(set(len(w) for w in words)) <= 2:
        return 0.2
    
    return 0.0


def pattern_similarity(word1: str, word2: str) -> float:
    """
    Compute pattern-based similarity between two words.
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        Combined pattern similarity score
    """
    scores = [
        detect_phrase_pattern(word1, word2),
        detect_compound_word_pattern(word1, word2),
        detect_homophone_pattern(word1, word2)
    ]
    
    # Return maximum pattern match (different patterns are mutually exclusive)
    return max(scores)


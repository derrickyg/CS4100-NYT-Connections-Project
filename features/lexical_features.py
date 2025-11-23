"""
Extract lexical features: edit distance, POS tags, prefixes, suffixes, etc.
"""
from typing import Tuple
import nltk
from nltk.corpus import wordnet

# Download required NLTK data
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


def edit_distance(word1: str, word2: str) -> int:
    """
    Compute Levenshtein edit distance between two words.
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        Edit distance
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def normalized_edit_distance(word1: str, word2: str) -> float:
    """
    Compute normalized edit distance (0 to 1 scale).
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        Normalized edit distance (0 = identical, 1 = completely different)
    """
    max_len = max(len(word1), len(word2))
    if max_len == 0:
        return 0.0
    return edit_distance(word1, word2) / max_len


def length_difference(word1: str, word2: str) -> int:
    """Compute absolute difference in word lengths."""
    return abs(len(word1) - len(word2))


def common_prefix_length(word1: str, word2: str) -> int:
    """Compute length of common prefix."""
    i = 0
    while i < len(word1) and i < len(word2) and word1[i] == word2[i]:
        i += 1
    return i


def common_suffix_length(word1: str, word2: str) -> int:
    """Compute length of common suffix."""
    i = 0
    while i < len(word1) and i < len(word2) and word1[-(i+1)] == word2[-(i+1)]:
        i += 1
    return i


def same_pos_tag(word1: str, word2: str) -> bool:
    """
    Check if two words have the same part of speech tag.
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        True if same POS tag
    """
    try:
        pos1 = nltk.pos_tag([word1])[0][1]
        pos2 = nltk.pos_tag([word2])[0][1]
        return pos1 == pos2
    except:
        return False


def lexical_similarity(word1: str, word2: str) -> float:
    """
    Compute combined lexical similarity score.
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        Similarity score between 0 and 1
    """
    # Normalized edit distance (inverted: lower distance = higher similarity)
    edit_sim = 1.0 - normalized_edit_distance(word1, word2)
    
    # Length similarity
    max_len = max(len(word1), len(word2))
    length_sim = 1.0 - (length_difference(word1, word2) / max_len) if max_len > 0 else 1.0
    
    # Prefix similarity
    max_len_pref = max(len(word1), len(word2))
    prefix_sim = common_prefix_length(word1, word2) / max_len_pref if max_len_pref > 0 else 0.0
    
    # Suffix similarity
    max_len_suff = max(len(word1), len(word2))
    suffix_sim = common_suffix_length(word1, word2) / max_len_suff if max_len_suff > 0 else 0.0
    
    # POS tag similarity
    pos_sim = 1.0 if same_pos_tag(word1, word2) else 0.0
    
    # Weighted combination
    similarity = (0.3 * edit_sim + 
                  0.2 * length_sim + 
                  0.2 * prefix_sim + 
                  0.2 * suffix_sim + 
                  0.1 * pos_sim)
    
    return similarity


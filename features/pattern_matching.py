"""
Detect common Connections patterns: "___ word", "word ___", compound words, etc.
"""
from typing import List

# Pattern similarity score constants
PHRASE_PATTERN_SCORE = 0.3  # Score for phrase patterns (e.g., "___ word")
COMPOUND_WORD_MIN_COMMON_LEN = 3  # Minimum common substring length for compound words
COMPOUND_WORD_MAX_SCORE = 0.5  # Maximum score for compound word patterns
COMPOUND_WORD_DIVISOR = 10.0  # Divisor for calculating compound word score
HOMOPHONE_LENGTH_DIFF_THRESHOLD = 1  # Maximum length difference for homophone detection
HOMOPHONE_LETTER_OVERLAP_THRESHOLD = 0.7  # Minimum letter overlap ratio for homophones
HOMOPHONE_PATTERN_SCORE = 0.4  # Score for homophone patterns
CATEGORY_LENGTH_VARIANCE_THRESHOLD = 2  # Maximum length variance for category pattern
CATEGORY_PATTERN_SCORE = 0.2  # Score for category patterns


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
        return PHRASE_PATTERN_SCORE
    
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
            for length in range(COMPOUND_WORD_MIN_COMMON_LEN, min(len(w1_lower) - i, len(w2_lower) - j) + 1):
                if w1_lower[i:i+length] == w2_lower[j:j+length]:
                    max_common_len = max(max_common_len, length)
    
    if max_common_len >= COMPOUND_WORD_MIN_COMMON_LEN:
        return min(max_common_len / COMPOUND_WORD_DIVISOR, COMPOUND_WORD_MAX_SCORE)
    
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
    if abs(len(word1) - len(word2)) <= HOMOPHONE_LENGTH_DIFF_THRESHOLD:
        # Count common letters
        common_letters = set(word1.lower()) & set(word2.lower())
        total_letters = set(word1.lower()) | set(word2.lower())
        
        if len(total_letters) > 0:
            similarity = len(common_letters) / len(total_letters)
            if similarity > HOMOPHONE_LETTER_OVERLAP_THRESHOLD:
                return HOMOPHONE_PATTERN_SCORE
    
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
    if len(set(len(w) for w in words)) <= CATEGORY_LENGTH_VARIANCE_THRESHOLD:
        return CATEGORY_PATTERN_SCORE
    
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


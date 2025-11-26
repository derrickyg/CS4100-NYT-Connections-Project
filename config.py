"""
Configuration parameters for NYT Connections Solver.
"""

# Similarity weights
EMBEDDING_WEIGHT = 0.7  # Semantic meaning from pre-trained embeddings
LEXICAL_WEIGHT = 0.2    # Edit distance, prefixes, suffixes
PATTERN_WEIGHT = 0.1    # Pattern matching

# Word embeddings
EMBEDDING_MODEL = "glove-wiki-gigaword-300"

# Iterative solver parameters
ITERATIVE_MAX_COMBINATIONS = 200  # Maximum number of word combinations to consider when finding most similar group
ITERATIVE_MAX_TO_CHECK = 100  # Maximum number of combinations to actually check for similarity (performance limit)
ITERATIVE_KNOWN_PAIR_BONUS = 2.0  # Bonus score added when a group contains a known pair (learned from feedback)

# CSP solver parameters
CONSISTENCY_THRESHOLD = 0.4  # Minimum average similarity for a group to be considered consistent   
ALMOST_FULL_THRESHOLD_MULTIPLIER = 0.7  # Multiplier to lower threshold when group is almost full
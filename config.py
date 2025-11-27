"""
Configuration parameters for NYT Connections Solver.
"""

# Word embeddings
EMBEDDING_MODEL = "glove-wiki-gigaword-300"

# Iterative solver parameters
ITERATIVE_MAX_COMBINATIONS = 200  # Maximum number of word combinations to consider when finding most similar group
ITERATIVE_MAX_TO_CHECK = 100  # Maximum number of combinations to actually check for similarity (performance limit)
ITERATIVE_KNOWN_PAIR_BONUS = 2.0  # Bonus score added when a group contains a known pair (learned from feedback)


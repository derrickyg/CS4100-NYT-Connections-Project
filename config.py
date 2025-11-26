"""
Configuration parameters for NYT Connections Solver.
"""

# play with these to see if better outcome can be achieved

# Similarity weights
EMBEDDING_WEIGHT = 0.7  # Semantic meaning from pre-trained embeddings
LEXICAL_WEIGHT = 0.2    # Edit distance, prefixes, suffixes
PATTERN_WEIGHT = 0.1    # Pattern matching

# CSP parameters
CONSISTENCY_THRESHOLD = 0.5  # minimum avg similarity for consistency
ALMOST_FULL_THRESHOLD_MULTIPLIER = 0.7  # multiplier for consistency threshold when group is almost full (3/4 words)

# Hill Climbing parameters
HC_NUM_RESTARTS = 10
HC_MAX_ITERATIONS = 1000

# Simulated Annealing parameters
SA_INITIAL_TEMP = 100
SA_COOLING_RATE = 0.95
SA_MAX_ITERATIONS = 1000
SA_MIN_TEMP = 0.01  # Minimum temperature threshold - algorithm stops when temperature drops below this

# Objective function weights
ALPHA = 1.0  # within-group similarity weight
BETA = 0.5   # between-group similarity weight

# ML model parameters
ML_TRAIN_TEST_SPLIT = 0.2
ML_CLASSIFIER = "logistic_regression"
ML_RANDOM_STATE = 42

# Word embeddings
EMBEDDING_MODEL = "glove-wiki-gigaword-300"

# Iterative solver parameters
ITERATIVE_MAX_COMBINATIONS = 200  # Maximum number of word combinations to consider when finding most similar group
ITERATIVE_MAX_TO_CHECK = 100  # Maximum number of combinations to actually check for similarity (performance limit)
ITERATIVE_KNOWN_PAIR_BONUS = 2.0  # Bonus score added when a group contains a known pair (learned from feedback)


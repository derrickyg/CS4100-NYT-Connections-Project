"""
Configuration parameters for NYT Connections Solver.
"""

# play with these to see if better outcome can be achieved

# Similarity weights
EMBEDDING_WEIGHT = 0.4
COOCCURRENCE_WEIGHT = 0.3
LEXICAL_WEIGHT = 0.2
PATTERN_WEIGHT = 0.1

# CSP parameters
CONSISTENCY_THRESHOLD = 0.5  # minimum avg similarity for consistency

# Hill Climbing parameters
HC_NUM_RESTARTS = 10
HC_MAX_ITERATIONS = 1000

# Simulated Annealing parameters
SA_INITIAL_TEMP = 100
SA_COOLING_RATE = 0.95
SA_MAX_ITERATIONS = 1000

# Objective function weights
ALPHA = 1.0  # within-group similarity weight
BETA = 0.5   # between-group similarity weight

# ML model parameters
ML_TRAIN_TEST_SPLIT = 0.2
ML_CLASSIFIER = "logistic_regression"
ML_RANDOM_STATE = 42

# Word embeddings
EMBEDDING_MODEL = "glove-wiki-gigaword-300"


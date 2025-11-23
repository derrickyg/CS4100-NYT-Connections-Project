# CS4100-NYT-Connections-Project

NYT Connections Solver Agent - An AI agent that solves NYT Connections puzzles using CSP formulation, local search algorithms, and multiple similarity metrics.

## Overview

This project implements a solver for NYT Connections puzzles following the Product Requirements Document. The solver uses:
- **CSP (Constraint Satisfaction Problem)** formulation with backtracking
- **Local search algorithms**: Hill Climbing with Random Restarts and Simulated Annealing
- **Multiple similarity metrics**: Word embeddings, co-occurrence statistics, lexical features, and pattern matching
- **Ensemble approach**: Combines multiple solvers to find the best solution

## Dataset

The solver uses the NYT Connections dataset from HuggingFace:
- Dataset: `tm21cy/NYT-Connections`
- Contains 652 puzzles with words, answers, and difficulty ratings

## Installation

1. Create a virtual environment (recommended):
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
   - On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```
   - On Windows:
   ```bash
   venv\Scripts\activate
   ```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (done automatically on first run, but can be done manually):
```python
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

## Project Structure

```
CS4100-NYT-Connections-Project/
├── data/
│   ├── load_dataset.py          # Load HuggingFace dataset
│   ├── preprocess.py             # Data preprocessing and validation
│   └── sample_puzzles.py        # Sample puzzles for testing
├── features/
│   ├── word_embeddings.py        # Load and use word vectors (GloVe)
│   ├── lexical_features.py       # Edit distance, POS tags, etc.
│   ├── cooccurrence.py           # Build co-occurrence statistics
│   └── pattern_matching.py       # Detect common patterns
├── similarity/
│   └── combined_similarity.py    # Weighted combination of similarity metrics
├── solvers/
│   ├── csp_solver.py             # CSP with backtracking
│   ├── hill_climbing.py          # Hill climbing with restarts
│   ├── simulated_annealing.py    # Simulated annealing
│   └── iterative_solver.py       # Iterative solver with feedback (game mode)
├── evaluation/
│   ├── metrics.py                # Accuracy metrics
│   ├── validator.py              # Solution validation
│   └── game_simulator.py         # Game simulation with feedback
├── config.py                     # Hyperparameters
├── main.py                       # Main solver orchestration
└── requirements.txt              # Python dependencies
```

## Usage

### Solve a Single Puzzle (Offline Mode)

Solve puzzle at index 0:
```bash
python main.py --mode single --index 0
```

Use single solver instead of ensemble:
```bash
python main.py --mode single --index 0 --no-ensemble
```

**Note**: In single mode, `--num-puzzles` is ignored. Use `--index` to select a specific puzzle.

### Evaluate on Multiple Puzzles (Offline Mode)

Evaluate on first 10 puzzles:
```bash
python main.py --mode evaluate --num-puzzles 10
```

Evaluate on all puzzles (may take a long time):
```bash
python main.py --mode evaluate --num-puzzles 652
```

### Game Simulation Mode (Follows Actual Game Rules)

Play a single puzzle with game rules (4 mistakes max by default):
```bash
python main.py --mode game --index 0
```

Play with custom number of mistakes allowed:
```bash
python main.py --mode game --index 0 --mistakes-allowed 7
```

Evaluate on multiple puzzles in game mode:
```bash
python main.py --mode game --num-puzzles 10
```

Use simpler sample puzzles (designed to be solvable):
```bash
python main.py --mode game --num-puzzles 10 --use-sample-games
```

**Game Mode Features:**
- Submits groups one at a time (like the real game)
- Receives feedback: correct/incorrect with partial match counts (e.g., "3/4 words correct")
- Tracks mistakes (configurable with `--mistakes-allowed`, default: 4)
- Learns from feedback to improve future guesses
- Avoids repeating previously tried groups
- Shows how many incorrect guesses it takes to solve each puzzle
- Displays accuracy metrics at the end of each game

**Sample Puzzles:**
- Use `--use-sample-games` flag to test with simpler puzzles
- Sample puzzles use clear semantic relationships (colors, animals, emotions, etc.)
- Designed to be solvable by word embeddings and similarity metrics
- Good for testing and demonstrating the solver's capabilities
- 10 sample puzzles available (difficulty 1.0-2.0 vs real puzzles 2.0-5.0)

## How It Works

### 1. Similarity Metrics

The solver combines multiple similarity measures:
- **Word Embeddings**: Pre-trained GloVe embeddings (`glove-wiki-gigaword-300`)
- **Co-occurrence**: Statistics from historical NYT Connections puzzles (uses PMI - Pointwise Mutual Information)
- **Lexical Features**: Edit distance, POS tags, prefixes, suffixes
- **Pattern Matching**: Phrase patterns, compound words, homophones (placeholder for future enhancement)

**Note**: When evaluating on test puzzles, those puzzles are automatically excluded from co-occurrence statistics to prevent data leakage.

### 2. CSP Solver

Formulates the puzzle as a Constraint Satisfaction Problem:
- Variables: 16 words
- Domains: {Group1, Group2, Group3, Group4}
- Constraints: Each group must contain exactly 4 words
- Uses backtracking with MRV (Minimum Remaining Values) heuristic and least constraining value ordering

### 3. Local Search Algorithms

- **Hill Climbing**: Maximizes within-group similarity, minimizes between-group similarity
- **Simulated Annealing**: Escapes local optima using temperature schedule

### 4. Ensemble Approach

Runs multiple solvers and selects the solution with the highest objective score.

### 5. Game Simulation Mode

Simulates the actual Connections game with feedback:
- Submits groups iteratively (one at a time)
- Receives feedback on each submission (correct/incorrect with partial matches like "3/4 words correct")
- Tracks mistakes (configurable max, default 4 before losing)
- Learns from feedback to improve future guesses:
  - Identifies known pairs from correct groups
  - Marks forbidden pairs from incorrect groups
  - Refines partial matches (e.g., tries variations of 3/4 correct groups)
- Avoids repeating previously tried groups
- Measures performance: how many incorrect guesses it takes to solve
- Uses fast similarity-based selection with random fallback (optimized for speed)

## Configuration

Edit `config.py` to adjust:
- Similarity metric weights
- CSP consistency threshold
- Hill Climbing restarts and iterations
- Simulated Annealing temperature and cooling rate
- Objective function weights

## Evaluation Metrics

The solver reports:
- **Partial Accuracy**: Percentage of correctly identified groups (0-100%)
- **Word Accuracy**: Percentage of words in correct groups
- **Exact Match**: Whether all 4 groups are correct

## Expected Performance

Based on the PRD success criteria:
- **Minimum Viable Product**: >25% partial accuracy (>1 correct group per puzzle)
- **Target Goals**: >40% partial accuracy (>1.6 correct groups per puzzle)
- **Stretch Goals**: >50% partial accuracy

## Notes

- **Initialization Time**: First run takes ~30-60 seconds to load word embeddings and build co-occurrence statistics. Subsequent runs are much faster due to caching.
- **Performance**: Game mode runs in seconds per puzzle (optimized for speed). Offline mode may take longer due to ensemble approach.
- **Data Leakage Prevention**: When evaluating on test puzzles, those puzzles are automatically excluded from co-occurrence statistics.
- **Solver Behavior**: 
  - The solver works best on puzzles with clear semantic relationships
  - Harder puzzles (Purple difficulty) may require cultural knowledge not captured in embeddings
  - CSP solver may fail on very difficult puzzles; local search provides fallback
  - Game mode uses iterative refinement strategies optimized for speed

## Future Enhancements

- Optional ML component: Train pairwise classifier on historical data
- Knowledge graph integration (ConceptNet)
- Category prediction for each group
- Interactive mode with human hints

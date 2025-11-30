# connections_solver/cli.py

from __future__ import annotations
import argparse
import random
from typing import List

from .puzzle import Puzzle
from .similarity import get_default_backend
from .local_search import hill_climbing, simulated_annealing


def print_partition(groups: List[List[str]]):
    for idx, group in enumerate(groups, start=1):
        print(f"Group {idx}: {group}")


def main():
    parser = argparse.ArgumentParser(
        description="NYT Connections solver using local search."
    )
    parser.add_argument("puzzle_path", type=str, help="Path to puzzle JSON file.")
    parser.add_argument(
        "--algo",
        type=str,
        default="hill",
        choices=["hill", "sa"],
        help="Which local search algorithm to use: hill (hill-climbing) or sa (simulated annealing).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=75,
        help="Number of random restarts (hill-climbing only).",
    )
    args = parser.parse_args()
    random.seed(args.seed)

    puzzle = Puzzle.from_json(args.puzzle_path)
    backend = get_default_backend()

    if args.algo == "hill":
        result = hill_climbing(
            puzzle.words,
            backend=backend,
            max_iterations=1000,
            restarts=args.restarts,
        )
    else:
        result = simulated_annealing(
            puzzle.words,
            backend=backend,
            max_iterations=5000,
            start_temperature=1.0,
            cooling_rate=0.0005,
        )

    print(f"Algorithm: {args.algo}")
    print(f"Score: {result.score:.4f}")
    print(f"Iterations: {result.iterations}")
    print(f"Restarts: {result.restarts}\n")

    print("Suggested grouping:")
    print_partition(result.groups)

    if puzzle.solution_groups:
        metrics = puzzle.evaluate_partition(result.groups)
        print("\nEvaluation vs true solution:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

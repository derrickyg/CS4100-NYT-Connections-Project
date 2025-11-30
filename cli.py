# cli.py

from __future__ import annotations
import argparse
import random
from typing import List
from pathlib import Path

from .puzzle import Puzzle


def random_partition(words: List[str], group_size: int = 4) -> List[List[str]]:
    shuffled = words[:]
    random.shuffle(shuffled)
    return [shuffled[i:i + group_size] for i in range(0, len(shuffled), group_size)]


def main():
    parser = argparse.ArgumentParser(
        description="NYT Connections solver (baseline random version)."
    )
    parser.add_argument(
        "puzzle_path",
        type=str,
        help="Path to puzzle JSON file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()
    random.seed(args.seed)

    puzzle = Puzzle.from_json(args.puzzle_path)
    partition = random_partition(puzzle.words)

    print("Random partition:")
    for idx, group in enumerate(partition, start=1):
        print(f"Group {idx}: {group}")

    if puzzle.solution_groups:
        metrics = puzzle.evaluate_partition(partition)
        print("\nEvaluation vs true solution:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

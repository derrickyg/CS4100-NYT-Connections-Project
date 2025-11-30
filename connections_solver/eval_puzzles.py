#!/usr/bin/env python3
"""
Run the Connections solver on every puzzle JSON in examples/
and report per-puzzle and aggregate performance.

Usage:
    python eval_all_puzzles.py --algo hill
    python eval_all_puzzles.py --algo sa --restarts 100
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Dict, Any

from connections_solver.puzzle import Puzzle
from connections_solver.similarity import get_default_backend
from connections_solver.local_search import hill_climbing, simulated_annealing


def eval_puzzle(
    puzzle_path: Path,
    algo: str,
    backend,
    restarts: int,
    max_iterations: int,
) -> Dict[str, Any]:
    """Run solver on a single puzzle and return metrics."""
    puzzle = Puzzle.from_json(puzzle_path)
    words = puzzle.words

    if algo == "hill":
        result = hill_climbing(
            words,
            backend=backend,
            max_iterations=max_iterations,
            restarts=restarts,
        )
    else:
        result = simulated_annealing(
            words,
            backend=backend,
            max_iterations=max_iterations,
            start_temperature=1.0,
            cooling_rate=0.0005,
        )

    metrics: Dict[str, Any] = {
        "puzzle": puzzle_path.name,
        "score": result.score,
        "iterations": result.iterations,
        "restarts": result.restarts,
    }

    if puzzle.solution_groups:
        eval_stats = puzzle.evaluate_partition(result.groups)
        metrics.update(eval_stats)
    else:
        # No ground truth available
        metrics["group_matches"] = None
        metrics["word_correct_group"] = None
        metrics["word_accuracy"] = None

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NYT Connections solver on all puzzles in examples/."
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default="examples",
        help="Directory containing puzzle JSON files.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="hill",
        choices=["hill", "sa"],
        help="Search algorithm: hill (hill-climbing) or sa (simulated annealing).",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=75,
        help="Number of random restarts (hill-climbing only).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Max iterations per run (hill-climbing) or total steps (SA).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    examples_dir = Path(args.examples_dir)
    if not examples_dir.exists():
        raise SystemExit(f"Examples directory not found: {examples_dir}")

    puzzle_paths: List[Path] = sorted(examples_dir.glob("*.json"))
    if not puzzle_paths:
        raise SystemExit(f"No .json puzzles found in {examples_dir}")

    backend = get_default_backend()

    print(f"Algorithm: {args.algo}")
    print(f"Examples directory: {examples_dir}")
    print(f"Found {len(puzzle_paths)} puzzle(s).\n")

    results: List[Dict[str, Any]] = []
    total_word_acc = 0.0
    total_group_matches = 0.0
    count_with_gt = 0

    for path in puzzle_paths:
        metrics = eval_puzzle(
            path,
            algo=args.algo,
            backend=backend,
            restarts=args.restarts,
            max_iterations=args.max_iterations,
        )
        results.append(metrics)

        has_gt = metrics["word_accuracy"] is not None
        if has_gt:
            count_with_gt += 1
            total_word_acc += metrics["word_accuracy"]
            total_group_matches += metrics["group_matches"]

        # Per-puzzle summary line
        wa = metrics["word_accuracy"]
        wa_str = f"{wa:.3f}" if wa is not None else "N/A"
        gm = metrics["group_matches"]
        gm_str = f"{int(gm)}" if gm is not None else "N/A"

        print(
            f"{metrics['puzzle']:<40} "
            f"score={metrics['score']:.4f}  "
            f"word_acc={wa_str}  "
            f"group_matches={gm_str}"
        )

    print("\n=== Aggregate over puzzles with ground truth ===")
    if count_with_gt == 0:
        print("No puzzles with solution groups; cannot compute aggregate accuracy.")
    else:
        avg_word_acc = total_word_acc / count_with_gt
        avg_group_matches = total_group_matches / count_with_gt
        print(f"Num puzzles with ground truth: {count_with_gt}")
        print(f"Avg word accuracy: {avg_word_acc:.3f}")
        print(f"Avg group matches: {avg_group_matches:.3f}")


if __name__ == "__main__":
    main()

"""
Main solver orchestration for game mode.
"""
import argparse
import time
from typing import List
from data.load_dataset import load_historical_data, Puzzle
from similarity.combined_similarity import CombinedSimilarity
from evaluation.game_simulator import GameSimulator
from solvers.iterative_solver import IterativeSolver


def solve_with_game_simulation(puzzle: Puzzle, similarity_fn: CombinedSimilarity, max_mistakes: int = 4):
    """
    Solve puzzle using game simulation with feedback.
    
    Args:
        puzzle: Puzzle to solve
        max_mistakes: Maximum number of mistakes allowed
        similarity_fn: Pre-initialized similarity function (optional, creates new if None)
        
    Returns:
        Dictionary with metrics
    """
    start_time = time.time()
    
    # Initialize game simulator
    game_init_start = time.time()
    game = GameSimulator(puzzle, max_mistakes=max_mistakes)
    game_init_time = time.time() - game_init_start
    
    # Initialize iterative solver (reuse similarity_fn if provided)
    solver_init_start = time.time()
    solver = IterativeSolver(similarity_fn)
    solver_init_time = time.time() - solver_init_start
    
    print(f"\npuzzle id: {puzzle.puzzle_id}")

    # Solve with feedback
    solve_start = time.time()
    result = solver.solve_with_feedback(game)
    solve_time = time.time() - solve_start
    
    total_time = time.time() - start_time
    
    # Display results
    num_correct = len(result['solved_groups'])
    print(f"Correct guesses: {num_correct}")
    
    # Add timing to result
    result['timing'] = {
        'game_init': game_init_time,
        'solver_init': solver_init_time,
        'solve': solve_time,
        'total': total_time
    }
    
    return result


def solve_puzzles(test_puzzles: List[Puzzle], max_mistakes: int = 4):
    """
    Solve a list of puzzles
    
    Args:
        test_puzzles: List of puzzles to solve
        max_mistakes: Maximum number of mistakes allowed per puzzle (default: 4)
    """
    results = []
    
    # initialize the similarity function
    similarity_fn = CombinedSimilarity()

    
    for puzzle in test_puzzles:
        result = solve_with_game_simulation(puzzle, similarity_fn, max_mistakes=max_mistakes)
        results.append(result)
    
    # Aggregate statistics
    total_puzzles = len(results)
    wins = sum(1 for r in results if r['is_won'])
    avg_correct = sum(len(r['solved_groups']) for r in results) / total_puzzles
        
    # Timing statistics
    avg_total_time = sum(r.get('timing', {}).get('total', 0) for r in results) / total_puzzles
    
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}")
    print(f"Wins: {wins} ({wins/total_puzzles:.1%})")
    print(f"Average correct guesses per puzzle: {avg_correct:.2f}")
    print(f"Average time per puzzle: {avg_total_time:.2f}s")
    print(f"{'='*70}\n")



def main():
    """let the agent play the game!"""
    parser = argparse.ArgumentParser(description="NYT Connections Solver Agent - Game Mode")
    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=None,
        help="Number of puzzles to solve (default: all)"
    )
    parser.add_argument(
        "--mistakes-allowed",
        type=int,
        default=4,
        help="Maximum number of mistakes allowed in game mode (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Load puzzles
    all_puzzles = load_historical_data()
    test_puzzles = all_puzzles if args.num_puzzles is None else all_puzzles[:args.num_puzzles]
    
    # solve the test puzzles
    solve_puzzles(test_puzzles, max_mistakes=args.mistakes_allowed)


if __name__ == "__main__":
    main()

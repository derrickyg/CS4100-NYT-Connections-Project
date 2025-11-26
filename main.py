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
    attempt to solve a puzzle and provide metrics on the model's performance
    
    Args:
        puzzle: Puzzle to solve
        max_mistakes: Maximum number of mistakes allowed
        similarity_fn: Pre-initialized similarity function
        
    Returns:
        Dictionary with metrics on the model's performance
    """
    start_time = time.time()
    
    # game simulator class - basically simulates the game
    game = GameSimulator(puzzle, max_mistakes=max_mistakes)
    
    # solver class - works with the GameSimulator state to make guesses to solve the puzzle
    solver = IterativeSolver(similarity_fn)
    
    print(f"\npuzzle id: {puzzle.puzzle_id}")

    # solve the puzzle using the IterativeSolver class on the GameSimulator state
    result = solver.solve_with_feedback(game)
    
    total_time = time.time() - start_time
    
    # display results
    num_correct = len(result['solved_groups'])
    print(f"Correct guesses: {num_correct}")
    
    result['timing'] = {'total': total_time}
    
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

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
from solvers.csp_solver import CSPSolver


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
    results_iterative = []
    results_csp = []
    
    # initialize the similarity function
    similarity_fn = CombinedSimilarity()

    def csp_groups_match_truth(csp_groups, truth_groups):
        if not csp_groups:
            return False
        # Convert both to sets of frozensets for order-insensitive comparison
        def group_set(groups):
            return set(frozenset([w.upper() for w in group]) for group in groups.values() if group)
        return group_set(csp_groups) == group_set(truth_groups)

    for puzzle in test_puzzles:
        result = solve_with_game_simulation(puzzle, similarity_fn, max_mistakes=max_mistakes)
        results_iterative.append(result)

        # --- CSP Solver statistics ---
        # Run CSP solver and time it
        start_time = time.time()
        csp_solver = CSPSolver(similarity_fn)
        solved_groups = []
        try:
            csp_groups = csp_solver.solve(puzzle.words)
            is_won = csp_groups_match_truth(csp_groups, puzzle.groups)
            if is_won:
                solved_groups.append(csp_groups)
            print(f"puzzle groups: {puzzle.groups}")
            print(f"csp groups: {csp_groups}")
            mistakes = 0  # CSP solver is not interactive, so mistakes are not tracked
        except Exception as e:
            print(f"CSP Solver failed with exception: {e}")
            csp_groups = {}
            is_won = False
            mistakes = None
        total_time_csp = time.time() - start_time
        results_csp.append({
            'solved_groups': solved_groups,
            'is_won': is_won,
            'mistakes': mistakes,
            'timing': {'total': total_time_csp}
        })
    
    # Aggregate statistics
    total_puzzles = len(results_iterative)
    wins_iterative = sum(1 for r in results_iterative if r['is_won'])
    avg_correct = sum(len(r['solved_groups']) for r in results_iterative) / total_puzzles
    
    wins_csp = sum(1 for r in results_csp if r['is_won'])
    avg_correct_csp = sum(len(r['solved_groups']) for r in results_csp) / total_puzzles
    avg_total_time = sum(r.get('timing', {}).get('total', 0) for r in results_iterative) / total_puzzles
    avg_total_time_csp = sum(r.get('timing', {}).get('total', 0) for r in results_csp) / total_puzzles
        
    # Timing statistics
    avg_total_time = sum(r.get('timing', {}).get('total', 0) for r in results_iterative) / total_puzzles
    avg_total_time_csp = sum(r.get('timing', {}).get('total', 0) for r in results_csp) / total_puzzles
    
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}")
    print("Iterative Solver Results:")
    print(f"Wins: {wins_iterative} ({wins_iterative/total_puzzles:.1%})")
    print(f"Average correct guesses per puzzle: {avg_correct:.2f}")
    print(f"Average time per puzzle: {avg_total_time:.2f}s")
    print(f"\nCSP Solver Results:")
    print(f"Wins: {wins_csp} ({wins_csp/total_puzzles:.1%})")
    print(f"Average correct guesses per puzzle: {avg_correct_csp:.2f}")
    print(f"Average time per puzzle: {avg_total_time_csp:.2f}s")

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

"""
Main solver orchestration for game mode.
"""
import argparse
import time
from typing import List, Dict
from solvers.k_means_solver import KMeansConnectionsSolver
from data.load_dataset import load_historical_data, Puzzle
from similarity.embedding_similarity import EmbeddingSimilarity
from evaluation.game_simulator import GameSimulator
from solvers.iterative_solver import IterativeSolver
from features.word_embeddings import WordEmbeddings
import config


def solve_with_game_simulation(puzzle: Puzzle, similarity_fn: EmbeddingSimilarity, max_mistakes: int = 4):
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

def solve_with_kmeans(puzzle: Puzzle, kmeans_solver: KMeansConnectionsSolver, 
                      max_mistakes: int = 4) -> Dict:
    """
    Solve puzzle using K-Means clustering approach.
    
    Args:
        puzzle: Puzzle to solve
        kmeans_solver: Pre-initialized K-Means solver
        max_mistakes: Maximum number of mistakes allowed
        
    Returns:
        Dictionary with metrics on the model's performance
    """
    start_time = time.time()
    
    # Get the 16 words from puzzle
    all_words = []
    for group_words in puzzle.groups.values():
        all_words.extend(group_words)
    
    # game simulator class
    game = GameSimulator(puzzle, max_mistakes=max_mistakes)
    
    print(f"\npuzzle id: {puzzle.puzzle_id}")
    
    # Get K-Means clustering
    clustering_start = time.time()
    predicted_groups = kmeans_solver.solve_constrained(all_words, n_init=1000)
    clustering_time = time.time() - clustering_start
    
    # Submit groups in order (sorted by cohesion)
    submissions = []
    for group in predicted_groups:
        if game.is_game_over:
            break
        
        feedback = game.submit_group(group)
        submissions.append({
            "group": group,
            "feedback": feedback
        })
        
        if feedback.is_correct:
            print(f"✓ Correct guess: {', '.join(group)}")
    
    total_time = time.time() - start_time
    
    # Get final state
    state = game.get_state()
    num_correct = len(game.get_solved_groups())
    print(f"Correct guesses: {num_correct}")
    
    return {
        "solved_groups": game.get_solved_groups(),
        "submissions": submissions,
        "total_submissions": len(submissions),
        "mistakes": state["mistakes"],
        "is_won": state["is_won"],
        "timing": {
            "total": total_time,
            "clustering": clustering_time
        }
    }


def solve_puzzles(test_puzzles: List[Puzzle], max_mistakes: int = 4, 
                 solver_type: str = "iterative"):
    """
    Solve a list of puzzles
    
    Args:
        test_puzzles: List of puzzles to solve
        max_mistakes: Maximum number of mistakes allowed per puzzle (default: 4)
        solver_type: Type of solver to use ("iterative" or "kmeans")
    """
    results = []
    
    # Initialize solver based on type
    if solver_type == "iterative":
        similarity_fn = EmbeddingSimilarity()
        print(f"\n{'='*70}")
        print("USING ITERATIVE SOLVER")
        print(f"{'='*70}\n")
        
        for puzzle in test_puzzles:
            result = solve_with_game_simulation(puzzle, similarity_fn, max_mistakes=max_mistakes)
            results.append(result)
    
    elif solver_type == "kmeans":
        embeddings_fn = WordEmbeddings(config.EMBEDDING_MODEL)
        
        kmeans_solver = KMeansConnectionsSolver(embeddings_fn)
        
        print(f"\n{'='*70}")
        print("USING K-MEANS CLUSTERING SOLVER")
        print(f"{'='*70}\n")
        
        for puzzle in test_puzzles:
            result = solve_with_kmeans(puzzle, kmeans_solver, max_mistakes=max_mistakes)
            results.append(result)
    
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
    
    # Aggregate statistics
    total_puzzles = len(results)
    wins = sum(1 for r in results if r['is_won'])
    avg_correct = sum(len(r['solved_groups']) for r in results) / total_puzzles
    avg_submissions = sum(r['total_submissions'] for r in results) / total_puzzles
    avg_mistakes = sum(r['mistakes'] for r in results) / total_puzzles
    
    # Timing statistics
    avg_total_time = sum(r.get('timing', {}).get('total', 0) for r in results) / total_puzzles
    
    print(f"\n{'='*70}")
    print(f"OVERALL STATISTICS - {solver_type.upper()} SOLVER")
    print(f"{'='*70}")
    print(f"Total puzzles: {total_puzzles}")
    print(f"Wins: {wins} ({wins/total_puzzles:.1%})")
    print(f"Average correct guesses per puzzle: {avg_correct:.2f}")
    print(f"Average submissions per puzzle: {avg_submissions:.2f}")
    print(f"Average mistakes per puzzle: {avg_mistakes:.2f}")
    print(f"Average time per puzzle: {avg_total_time:.2f}s")
    
    if solver_type == "kmeans":
        avg_clustering_time = sum(r.get('timing', {}).get('clustering', 0) for r in results) / total_puzzles
        print(f"Average clustering time: {avg_clustering_time:.2f}s")
    
    print(f"{'='*70}\n")
    
    return results

def compare_solvers(test_puzzles: List[Puzzle], max_mistakes: int = 4):
    """
    Compare iterative and K-Means solvers side by side.
    
    Args:
        test_puzzles: List of puzzles to solve
        max_mistakes: Maximum number of mistakes allowed per puzzle
    """
    print(f"\n{'='*70}")
    print("COMPARING SOLVERS")
    print(f"{'='*70}\n")
    
    # Run iterative solver
    print("Running Iterative Solver...")
    iterative_results = solve_puzzles(test_puzzles, max_mistakes, solver_type="iterative")
    
    # Run K-Means solver
    print("\nRunning K-Means Solver...")
    kmeans_results = solve_puzzles(test_puzzles, max_mistakes, solver_type="kmeans")
    
    # Side-by-side comparison
    print(f"\n{'='*70}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Iterative':<20} {'K-Means':<20}")
    print(f"{'-'*70}")
    
    total_puzzles = len(test_puzzles)
    
    # Wins
    iter_wins = sum(1 for r in iterative_results if r['is_won'])
    kmeans_wins = sum(1 for r in kmeans_results if r['is_won'])
    print(f"{'Win Rate':<30} {iter_wins/total_puzzles:<20.1%} {kmeans_wins/total_puzzles:<20.1%}")
    
    # Average correct
    iter_correct = sum(len(r['solved_groups']) for r in iterative_results) / total_puzzles
    kmeans_correct = sum(len(r['solved_groups']) for r in kmeans_results) / total_puzzles
    print(f"{'Avg Correct Guesses':<30} {iter_correct:<20.2f} {kmeans_correct:<20.2f}")
    
    # Average submissions
    iter_subs = sum(r['total_submissions'] for r in iterative_results) / total_puzzles
    kmeans_subs = sum(r['total_submissions'] for r in kmeans_results) / total_puzzles
    print(f"{'Avg Submissions':<30} {iter_subs:<20.2f} {kmeans_subs:<20.2f}")
    
    # Average time
    iter_time = sum(r.get('timing', {}).get('total', 0) for r in iterative_results) / total_puzzles
    kmeans_time = sum(r.get('timing', {}).get('total', 0) for r in kmeans_results) / total_puzzles
    print(f"{'Avg Time (seconds)':<30} {iter_time:<20.2f} {kmeans_time:<20.2f}")
    
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
    parser.add_argument(
        "--solver-type",
        type=str,
        default="iterative",
        help="Type of solver to use (default: iterative)"
    )
    
    args = parser.parse_args()
    
    # Load puzzles
    all_puzzles = load_historical_data()
    test_puzzles = all_puzzles if args.num_puzzles is None else all_puzzles[:args.num_puzzles]
    
    # solve the test puzzles
    solve_puzzles(test_puzzles, max_mistakes=args.mistakes_allowed, solver_type=args.solver_type)


if __name__ == "__main__":
    main()

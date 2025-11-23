"""
Main solver orchestration with ensemble approach.
"""
import argparse
import time
from typing import List, Dict, Optional
from data.load_dataset import load_test_puzzle, load_historical_data, Puzzle
from data.sample_puzzles import get_sample_puzzles, get_sample_puzzle_by_index
from solvers.csp_solver import CSPSolver
from solvers.hill_climbing import HillClimbingSolver
from solvers.simulated_annealing import SimulatedAnnealingSolver
from similarity.combined_similarity import CombinedSimilarity
from evaluation.validator import validate_solution, compute_objective_score
from evaluation.metrics import compute_accuracy, compute_word_accuracy, compute_exact_match
from evaluation.game_simulator import GameSimulator
from solvers.iterative_solver import IterativeSolver


def solve_puzzle(words: List[str], use_ensemble: bool = True) -> Dict[int, List[str]]:
    """
    Main function to solve a NYT Connections puzzle.
    
    Args:
        words: List of 16 words
        use_ensemble: If True, run multiple solvers and combine results
    
    Returns:
        Dictionary mapping group_id to list of words in that group
    """
    # Initialize similarity function
    # Note: First initialization loads models (takes ~30-60s), subsequent uses are cached
    print("Initializing similarity function...")
    similarity_fn = CombinedSimilarity()
    
    if use_ensemble:
        # Run multiple solvers
        solutions = []
        
        # CSP Solver
        print("Running CSP Solver...")
        try:
            csp_solver = CSPSolver(similarity_fn)
            csp_solution = csp_solver.solve(words)
            if csp_solution and validate_solution(csp_solution):
                solutions.append(('CSP', csp_solution))
                print("  CSP found constraint-satisfying grouping (4 groups of 4 words)")
            else:
                print("  CSP solver did not find grouping (consistency threshold too strict)")
        except Exception as e:
            print(f"  CSP solver error: {e}")
        
        # Hill Climbing
        print("Running Hill Climbing...")
        try:
            hc_solver = HillClimbingSolver(similarity_fn)
            hc_solution = hc_solver.solve(words)
            if validate_solution(hc_solution):
                solutions.append(('HillClimbing', hc_solution))
                print("  Hill Climbing found constraint-satisfying grouping (4 groups of 4 words)")
            else:
                print("  Hill Climbing did not find valid grouping (should not happen)")
        except Exception as e:
            print(f"  Hill Climbing error: {e}")
        
        # Simulated Annealing
        print("Running Simulated Annealing...")
        try:
            sa_solver = SimulatedAnnealingSolver(similarity_fn)
            sa_solution = sa_solver.solve(words)
            if validate_solution(sa_solution):
                solutions.append(('SimulatedAnnealing', sa_solution))
                print("  Simulated Annealing found constraint-satisfying grouping (4 groups of 4 words)")
            else:
                print("  Simulated Annealing did not find valid grouping (should not happen)")
        except Exception as e:
            print(f"  Simulated Annealing error: {e}")
        
        # Select best solution based on objective score
        if solutions:
            best_solution = max(solutions, 
                              key=lambda x: compute_objective_score(x[1], similarity_fn))
            print(f"\nEnsemble Result: Selected best grouping from {best_solution[0]}")
            print("  (based on objective score: within-group similarity - between-group similarity)")
            print("  Note: This satisfies structural constraints (4 groups of 4) but may not be correct.")
            return best_solution[1]
        else:
            print("No valid groupings found, using Hill Climbing as fallback")
            hc_solver = HillClimbingSolver(similarity_fn)
            return hc_solver.solve(words)
    
    else:
        # Use single solver (CSP by default)
        solver = CSPSolver(similarity_fn)
        solution = solver.solve(words)
        if solution:
            return solution
        else:
            # Fallback to hill climbing
            print("CSP failed, falling back to Hill Climbing...")
            hc_solver = HillClimbingSolver(similarity_fn)
            return hc_solver.solve(words)


def evaluate_on_test_set(test_puzzles: List[Puzzle]):
    """
    Evaluate solver on multiple test puzzles.
    
    Args:
        test_puzzles: List of test puzzles
    """
    total_accuracy = 0.0
    total_word_accuracy = 0.0
    exact_matches = 0
    num_puzzles = len(test_puzzles)
    
    for i, puzzle in enumerate(test_puzzles):
        print(f"\nSolving puzzle {i+1}/{num_puzzles}: {puzzle.puzzle_id}")
        if puzzle.contest:
            print(f"  Contest: {puzzle.contest}")
        
        try:
            predicted_groups = solve_puzzle(puzzle.words, use_ensemble=True)
            
            if puzzle.groups:
                accuracy = compute_accuracy(predicted_groups, puzzle.groups)
                word_accuracy = compute_word_accuracy(predicted_groups, puzzle.groups)
                exact = compute_exact_match(predicted_groups, puzzle.groups)
                
                total_accuracy += accuracy
                total_word_accuracy += word_accuracy
                if exact:
                    exact_matches += 1
                
                print(f"  Partial Accuracy: {accuracy:.2%}")
                print(f"  Word Accuracy: {word_accuracy:.2%}")
                print(f"  Exact Match: {'Yes' if exact else 'No'}")
            else:
                print("  No ground truth available")
        except Exception as e:
            print(f"  Error solving puzzle: {e}")
    
    if num_puzzles > 0:
        avg_accuracy = total_accuracy / num_puzzles
        avg_word_accuracy = total_word_accuracy / num_puzzles
        exact_match_rate = exact_matches / num_puzzles
        
        print("\n" + "="*50)
        print("OVERALL RESULTS")
        print("="*50)
        print(f"Average Partial Accuracy: {avg_accuracy:.2%}")
        print(f"Average Word Accuracy: {avg_word_accuracy:.2%}")
        print(f"Exact Match Rate: {exact_match_rate:.2%}")
        print("="*50)
        
        return avg_accuracy


def solve_with_game_simulation(puzzle: Puzzle, verbose: bool = True, max_mistakes: int = 4, 
                              similarity_fn: Optional[CombinedSimilarity] = None):
    """
    Solve puzzle using game simulation with feedback.
    
    Args:
        puzzle: Puzzle to solve
        verbose: Whether to print detailed output
        max_mistakes: Maximum number of mistakes allowed
        similarity_fn: Pre-initialized similarity function (optional, creates new if None)
        
    Returns:
        Dictionary with results
    """
    start_time = time.time()
    
    if verbose:
        print(f"\n{'='*60}")
        print("GAME SIMULATION MODE")
        print(f"{'='*60}")
        print(f"Puzzle: {puzzle.contest or puzzle.puzzle_id}")
        if puzzle.difficulty:
            print(f"Difficulty: {puzzle.difficulty}")
        print(f"\nWords: {', '.join(puzzle.words)}")
        print(f"\nRules: Submit groups of 4 words. Max {max_mistakes} mistakes allowed.")
        print(f"{'='*60}\n")
    
    # Initialize game simulator
    game_init_start = time.time()
    game = GameSimulator(puzzle, max_mistakes=max_mistakes)
    game_init_time = time.time() - game_init_start
    
    # Initialize iterative solver (reuse similarity_fn if provided)
    solver_init_start = time.time()
    if similarity_fn is None:
        similarity_fn = CombinedSimilarity()
    solver = IterativeSolver(similarity_fn)
    solver_init_time = time.time() - solver_init_start
    
    # Solve with feedback
    solve_start = time.time()
    result = solver.solve_with_feedback(game)
    solve_time = time.time() - solve_start
    
    total_time = time.time() - start_time
    
    # Compute accuracy metrics
    if puzzle.groups:
        # Convert solved groups to same format as ground truth
        solved_groups_dict = result['solved_groups']
        if len(solved_groups_dict) > 0:
            # Reconstruct full solution for accuracy calculation
            # If game was won, we have all 4 groups
            # If lost, we only have partial groups
            full_solution = {}
            remaining_words = set(w.upper() for w in puzzle.words)
            
            # Add solved groups
            for group_id, words in solved_groups_dict.items():
                full_solution[group_id] = words
                remaining_words -= set(w.upper() for w in words)
            
            # Add remaining words to a dummy group for accuracy calculation
            if remaining_words:
                next_group_id = max(full_solution.keys()) + 1 if full_solution else 1
                full_solution[next_group_id] = list(remaining_words)
            
            accuracy = compute_accuracy(full_solution, puzzle.groups)
            word_accuracy = compute_word_accuracy(full_solution, puzzle.groups)
        else:
            # No groups solved - calculate accuracy from best partial matches
            # Find the best submission that had partial matches
            best_submission_accuracy = 0.0
            best_word_accuracy = 0.0
            
            for submission in result['submissions']:
                if submission['feedback'].correct_words > 0:
                    # Create a temporary solution with this submission as one group
                    # NOTE: This creates groups with only 1 word each for remaining words,
                    # which violates the game constraint (groups must have 4 words).
                    # This is a workaround ONLY for partial accuracy calculation purposes.
                    # The accuracy functions can handle this structure, but this should
                    # NOT be used for actual game validation or solution validation.
                    temp_solution = {1: submission['group']}
                    remaining = set(w.upper() for w in puzzle.words) - set(w.upper() for w in submission['group'])
                    
                    # Distribute remaining words into dummy groups (1 word each)
                    # This is only for accuracy calculation, not a valid game solution
                    group_id = 2
                    for word in remaining:
                        temp_solution[group_id] = [word]
                        group_id += 1
                    
                    temp_acc = compute_accuracy(temp_solution, puzzle.groups)
                    temp_word_acc = compute_word_accuracy(temp_solution, puzzle.groups)
                    
                    if temp_acc > best_submission_accuracy:
                        best_submission_accuracy = temp_acc
                    if temp_word_acc > best_word_accuracy:
                        best_word_accuracy = temp_word_acc
            
            accuracy = best_submission_accuracy
            word_accuracy = best_word_accuracy
    else:
        accuracy = None
        word_accuracy = None
    
    # Display results
    if verbose:
        print(f"\n{'='*60}")
        print("GAME RESULTS")
        print(f"{'='*60}")
        print(f"Total Submissions: {result['total_submissions']}")
        print(f"Mistakes: {result['mistakes']}/{max_mistakes}")
        print(f"Result: {'WON!' if result['is_won'] else f'LOST ({max_mistakes} mistakes)'}")
        
        if accuracy is not None:
            print(f"\nAccuracy Metrics:")
            print(f"  Partial Accuracy: {accuracy:.2%}")
            print(f"  Word Accuracy: {word_accuracy:.2%}")
            
            # Show best partial match from submissions
            best_partial = max(
                (s['feedback'].correct_words for s in result['submissions'] if not s['feedback'].is_correct),
                default=0
            )
            if best_partial > 0:
                print(f"  Best Partial Match: {best_partial}/4 words correct")
        
        if result['is_won']:
            print(f"\nSolved Groups:")
            for group_id in sorted(result['solved_groups'].keys()):
                desc = puzzle.category_descriptions.get(group_id, "")
                print(f"  Group {group_id} ({desc}): {', '.join(result['solved_groups'][group_id])}")
        else:
            # Show solved groups if any
            if len(result['solved_groups']) > 0:
                print(f"\nPartially Solved Groups (before losing):")
                for group_id in sorted(result['solved_groups'].keys()):
                    desc = puzzle.category_descriptions.get(group_id, "")
                    print(f"  Group {group_id} ({desc}): {', '.join(result['solved_groups'][group_id])}")
            else:
                print(f"\nNo groups were fully solved.")
            
            # Show best attempts
            print(f"\nBest Attempts:")
            submissions_with_feedback = [
                s for s in result['submissions'] 
                if s['feedback'].correct_words > 0
            ]
            if submissions_with_feedback:
                # Sort by correct words (descending)
                submissions_with_feedback.sort(
                    key=lambda x: x['feedback'].correct_words, 
                    reverse=True
                )
                for i, sub in enumerate(submissions_with_feedback[:3], 1):  # Show top 3
                    print(f"  Attempt {i}: {', '.join(sub['group'])} "
                          f"({sub['feedback'].correct_words}/4 correct)")
            else:
                print(f"  No partial matches found.")
        
        print(f"\n{'='*60}")
        print(f"\nTiming Information:")
        print(f"  Game initialization: {game_init_time:.2f}s")
        print(f"  Solver initialization: {solver_init_time:.2f}s")
        print(f"  Solving time: {solve_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"{'='*60}")
    
    # Add timing to result
    result['timing'] = {
        'game_init': game_init_time,
        'solver_init': solver_init_time,
        'solve': solve_time,
        'total': total_time
    }
    
    return result


def evaluate_game_mode(test_puzzles: List[Puzzle], max_mistakes: int = 4, exclude_indices: Optional[List[int]] = None):
    """
    Evaluate solver in game mode on multiple puzzles.
    
    Args:
        test_puzzles: List of puzzles to test
        max_mistakes: Maximum number of mistakes allowed per puzzle
        exclude_indices: Optional list of puzzle indices to exclude from co-occurrence stats
                        (prevents data leakage)
    """
    results = []
    
    # Initialize solver once for all puzzles (saves time)
    # Models are cached globally, so subsequent initializations are fast
    print("Initializing solver (this may take 30-60 seconds on first run due to model loading)...")
    solver_init_start = time.time()
    similarity_fn = CombinedSimilarity(exclude_puzzle_indices=exclude_indices)
    solver_init_time = time.time() - solver_init_start
    print(f"Solver initialized in {solver_init_time:.2f}s. Models are cached for future use.\n")
    
    for i, puzzle in enumerate(test_puzzles):
        print(f"\n{'='*70}")
        print(f"Puzzle {i+1}/{len(test_puzzles)}: {puzzle.puzzle_id}")
        print(f"{'='*70}")
        
        result = solve_with_game_simulation(puzzle, verbose=True, max_mistakes=max_mistakes, 
                                          similarity_fn=similarity_fn)
        results.append(result)
    
    # Aggregate statistics
    total_puzzles = len(results)
    wins = sum(1 for r in results if r['is_won'])
    losses = total_puzzles - wins
    
    avg_submissions = sum(r['total_submissions'] for r in results) / total_puzzles
    avg_mistakes = sum(r['mistakes'] for r in results) / total_puzzles
    
    win_rate_submissions = sum(r['total_submissions'] for r in results if r['is_won']) / wins if wins > 0 else 0
    win_rate_mistakes = sum(r['mistakes'] for r in results if r['is_won']) / wins if wins > 0 else 0
    
    # Timing statistics
    avg_total_time = sum(r.get('timing', {}).get('total', 0) for r in results) / total_puzzles
    avg_solve_time = sum(r.get('timing', {}).get('solve', 0) for r in results) / total_puzzles
    avg_init_time = sum(r.get('timing', {}).get('solver_init', 0) for r in results) / total_puzzles
    
    print(f"\n{'='*70}")
    print("OVERALL GAME MODE STATISTICS")
    print(f"{'='*70}")
    print(f"Total Puzzles: {total_puzzles}")
    print(f"Wins: {wins} ({wins/total_puzzles:.1%})")
    print(f"Losses: {losses} ({losses/total_puzzles:.1%})")
    print(f"\nAverage Submissions (all): {avg_submissions:.1f}")
    print(f"Average Mistakes (all): {avg_mistakes:.1f}")
    if wins > 0:
        print(f"\nAverage Submissions (wins only): {win_rate_submissions:.1f}")
        print(f"Average Mistakes (wins only): {win_rate_mistakes:.1f}")
    
    print(f"\nTiming Statistics:")
    print(f"  Average initialization time: {avg_init_time:.2f}s")
    print(f"  Average solving time: {avg_solve_time:.2f}s")
    print(f"  Average total time: {avg_total_time:.2f}s")
    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NYT Connections Solver Agent")
    parser.add_argument(
        "--mode",
        choices=["single", "evaluate", "game"],
        default="single",
        help="Mode: solve single puzzle, evaluate on dataset, or game simulation"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Puzzle index (for single mode)"
    )
    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=None,
        help="Number of puzzles to evaluate (for evaluate/game mode). If not set, evaluates single puzzle."
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Use single solver instead of ensemble"
    )
    parser.add_argument(
        "--mistakes-allowed",
        type=int,
        default=4,
        help="Maximum number of mistakes allowed in game mode (default: 4)"
    )
    parser.add_argument(
        "--use-sample-games",
        action="store_true",
        help="Use simpler sample puzzles instead of real NYT Connections puzzles"
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        # Load test puzzle
        puzzle = load_test_puzzle(args.index)
        
        print(f"Solving puzzle #{args.index}: {puzzle.contest or puzzle.puzzle_id}")
        if puzzle.difficulty:
            print(f"Difficulty: {puzzle.difficulty}")
        print(f"\nWords: {', '.join(puzzle.words)}")
        
        # Solve
        solution = solve_puzzle(puzzle.words, use_ensemble=not args.no_ensemble)
        
        # Display solution
        print("\n=== Solution ===")
        for group_id in sorted(solution.keys()):
            print(f"Group {group_id}: {', '.join(solution[group_id])}")
        
        # If ground truth available, compute accuracy
        if puzzle.groups:
            accuracy = compute_accuracy(solution, puzzle.groups)
            word_accuracy = compute_word_accuracy(solution, puzzle.groups)
            exact = compute_exact_match(solution, puzzle.groups)
            
            print("\n=== Evaluation ===")
            print(f"Partial Accuracy: {accuracy:.2%}")
            print(f"Word Accuracy: {word_accuracy:.2%}")
            print(f"Exact Match: {'Yes' if exact else 'No'}")
            
            print("\n=== Ground Truth ===")
            for group_id in sorted(puzzle.groups.keys()):
                desc = puzzle.category_descriptions.get(group_id, "")
                print(f"Group {group_id} ({desc}): {', '.join(puzzle.groups[group_id])}")
    
    elif args.mode == "evaluate":
        # Load puzzles
        all_puzzles = load_historical_data()
        num_puzzles = args.num_puzzles if args.num_puzzles is not None else 10
        test_puzzles = all_puzzles[:num_puzzles]
        
        evaluate_on_test_set(test_puzzles)
    
    elif args.mode == "game":
        # Load puzzle(s) for game simulation
        max_mistakes = args.mistakes_allowed
        
        if args.use_sample_games:
            # Use sample puzzles
            sample_puzzles = get_sample_puzzles()
            
            if args.num_puzzles is not None and args.num_puzzles > 1:
                # Evaluate on multiple sample puzzles
                num_puzzles = min(args.num_puzzles, len(sample_puzzles))
                test_puzzles = sample_puzzles[:num_puzzles]
                print(f"Using {num_puzzles} sample puzzles (simpler, designed to be solvable)")
                evaluate_game_mode(test_puzzles, max_mistakes=max_mistakes)
            else:
                # Single sample puzzle
                puzzle = get_sample_puzzle_by_index(args.index)
                solve_with_game_simulation(puzzle, max_mistakes=max_mistakes)
        else:
            # Use real NYT Connections puzzles
            if args.num_puzzles is not None and args.num_puzzles > 1:
                # Evaluate on multiple puzzles
                all_puzzles = load_historical_data()
                test_puzzles = all_puzzles[:args.num_puzzles]
                # Exclude test puzzle indices from co-occurrence stats to prevent data leakage
                test_indices = list(range(args.num_puzzles))
                print(f"⚠️  Note: Excluding puzzles {test_indices} from co-occurrence stats to prevent data leakage")
                evaluate_game_mode(test_puzzles, max_mistakes=max_mistakes, exclude_indices=test_indices)
            else:
                # Single puzzle
                puzzle = load_test_puzzle(args.index)
                
                if not puzzle.groups:
                    print("Error: Puzzle must have ground truth groups for game simulation.")
                    return
                
                solve_with_game_simulation(puzzle, max_mistakes=max_mistakes)


if __name__ == "__main__":
    main()

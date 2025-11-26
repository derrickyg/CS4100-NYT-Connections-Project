"""
Main solver orchestration for game mode.
"""
import argparse
import time
from typing import List, Dict, Optional
from data.load_dataset import load_historical_data, Puzzle
from similarity.combined_similarity import CombinedSimilarity
from evaluation.metrics import compute_accuracy, compute_word_accuracy
from evaluation.game_simulator import GameSimulator
from solvers.iterative_solver import IterativeSolver


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
        # Always load models fresh to prevent data leakage
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


def solve_puzzles(test_puzzles: List[Puzzle], max_mistakes: int = 4):
    """
    Solve a list of puzzles
    
    Args:
        test_puzzles: List of puzzles to solve
        max_mistakes: Maximum number of mistakes allowed per puzzle (default: 4)
    """
    results = []
    
    # initialize the similarity function (solver)
    similarity_fn = CombinedSimilarity()

    
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
    """let the agent play the game!"""
    parser = argparse.ArgumentParser(description="NYT Connections Solver Agent - Game Mode")
    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=1,
        help="Number of puzzles to solve (default: 1)"
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
    test_puzzles = all_puzzles[:args.num_puzzles]
    
    # solve the test puzzles
    solve_puzzles(test_puzzles, max_mistakes=args.mistakes_allowed)


if __name__ == "__main__":
    main()

"""
Random Solver for NYT Connections puzzles.
Simulates the guessing process and respects the mistakes allowed.
"""
from typing import List, Dict
from evaluation.game_simulator import GameSimulator
import random

class RandomSolver:
    def solve_with_feedback(self, game: GameSimulator) -> Dict:
        words = game.get_remaining_words()
        submissions = []
        used_groups = set()
        while not game.is_game_over:
            remaining = game.get_remaining_words()
            if len(remaining) < 4:
                break
            # Randomly select a group of 4 words that hasn't been tried
            max_attempts = 100
            for _ in range(max_attempts):
                group = random.sample(remaining, 4)
                group_tuple = tuple(sorted(w.upper() for w in group))
                if group_tuple not in used_groups:
                    used_groups.add(group_tuple)
                    break
            else:
                # Fallback: just take the first 4
                group = remaining[:4]
            feedback = game.submit_group(group)
            submissions.append({"group": group, "feedback": feedback})
        state = game.get_state()
        return {
            "solved_groups": game.get_solved_groups(),
            "submissions": submissions,
            "total_submissions": len(submissions),
            "mistakes": state["mistakes"],
            "is_won": state["is_won"]
        }

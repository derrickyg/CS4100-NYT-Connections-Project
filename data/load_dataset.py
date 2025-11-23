"""
Load and parse historical puzzle data from HuggingFace dataset.
"""
from datasets import load_dataset
from typing import List, Dict, Optional
from data.preprocess import preprocess_puzzle, validate_puzzle


class Puzzle:
    """Represents a NYT Connections puzzle."""
    
    def __init__(self, puzzle_id: str, words: List[str], 
                 groups: Optional[Dict[int, List[str]]] = None,
                 category_descriptions: Optional[Dict[int, str]] = None,
                 difficulty: Optional[float] = None,
                 contest: Optional[str] = None,
                 date: Optional[str] = None):
        self.puzzle_id = puzzle_id
        self.words = words
        self.groups = groups or {}
        self.category_descriptions = category_descriptions or {}
        self.difficulty = difficulty
        self.contest = contest
        self.date = date


def load_nyt_connections_dataset():
    """Load the NYT Connections dataset from HuggingFace."""
    dataset = load_dataset("tm21cy/NYT-Connections", split="train")
    return dataset


def parse_puzzle_from_dataset(dataset_item, index: int) -> Puzzle:
    """
    Parse a puzzle from the HuggingFace dataset format.
    
    Args:
        dataset_item: Single item from the dataset
        index: Index of the puzzle
        
    Returns:
        Puzzle object
    """
    words = dataset_item["words"]
    
    # Parse answers into groups
    groups = {}
    category_descriptions = {}
    
    if "answers" in dataset_item:
        for i, answer in enumerate(dataset_item["answers"], 1):
            groups[i] = answer["words"]
            category_descriptions[i] = answer.get("answerDescription", "")
    
    puzzle_id = f"puzzle_{index}"
    
    puzzle = Puzzle(
        puzzle_id=puzzle_id,
        words=words,
        groups=groups,
        category_descriptions=category_descriptions,
        difficulty=dataset_item.get("difficulty"),
        contest=dataset_item.get("contest"),
        date=str(dataset_item.get("date", ""))
    )
    
    # Preprocess and validate the puzzle
    puzzle = preprocess_puzzle(puzzle)
    
    return puzzle


def load_historical_data() -> List[Puzzle]:
    """
    Load all historical puzzles from the dataset.
    Puzzles are automatically preprocessed and validated.
    """
    dataset = load_nyt_connections_dataset()
    puzzles = []
    
    for i in range(len(dataset)):
        puzzle = parse_puzzle_from_dataset(dataset[i], i)
        # Validate puzzle structure
        if validate_puzzle(puzzle):
            puzzles.append(puzzle)
        else:
            print(f"Warning: Puzzle {i} failed validation, skipping")
    
    return puzzles


def load_test_puzzle(index: int = 0) -> Puzzle:
    """Load a specific puzzle by index for testing."""
    dataset = load_nyt_connections_dataset()
    return parse_puzzle_from_dataset(dataset[index], index)


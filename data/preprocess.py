"""
Parse and clean puzzle data from the HuggingFace dataset.
Provides utilities for data preprocessing and building training datasets.
"""
from typing import List, Dict, Tuple, TYPE_CHECKING
from itertools import combinations

# Avoid circular import by using TYPE_CHECKING
if TYPE_CHECKING:
    from data.load_dataset import Puzzle


def clean_word(word: str) -> str:
    """
    Clean and normalize a word.
    
    Args:
        word: Raw word string
        
    Returns:
        Cleaned word (uppercase, stripped)
    """
    return word.strip().upper()


def clean_puzzle_words(words: List[str]) -> List[str]:
    """
    Clean all words in a puzzle.
    
    Args:
        words: List of raw words
        
    Returns:
        List of cleaned words
    """
    return [clean_word(word) for word in words]


def validate_puzzle(puzzle: 'Puzzle') -> bool:
    """
    Validate that a puzzle has the correct structure.
    
    Args:
        puzzle: Puzzle to validate
        
    Returns:
        True if valid
    """
    # Check that puzzle has exactly 16 words
    if len(puzzle.words) != 16:
        return False
    
    # Check that all words are unique
    if len(set(puzzle.words)) != 16:
        return False
    
    # If groups are provided, validate them
    if puzzle.groups:
        if len(puzzle.groups) != 4:
            return False
        
        # Check each group has 4 words
        all_group_words = []
        for group_words in puzzle.groups.values():
            if len(group_words) != 4:
                return False
            all_group_words.extend(group_words)
        
        # Check all words are accounted for
        if set(puzzle.words) != set(all_group_words):
            return False
    
    return True


def build_word_pair_dataset(puzzles: List['Puzzle']) -> Tuple[List[Dict], List[int]]:
    """
    Build training dataset of word pairs with labels.
    Used for optional ML model training.
    
    Args:
        puzzles: List of puzzles with ground truth groups
        
    Returns:
        Tuple of (features_list, labels_list)
        - features_list: List of feature dictionaries for each word pair
        - labels_list: List of labels (1 if same group, 0 if different groups)
    """
    features_list = []
    labels_list = []
    
    for puzzle in puzzles:
        if not puzzle.groups:
            continue  # Skip puzzles without ground truth
        
        # Build mapping of word to group
        word_to_group = {}
        for group_id, words in puzzle.groups.items():
            for word in words:
                word_to_group[word.upper()] = group_id
        
        # Generate positive examples (word pairs in same group)
        for group_words in puzzle.groups.values():
            for word1, word2 in combinations(group_words, 2):
                features = {
                    'word1': word1.upper(),
                    'word2': word2.upper(),
                    'puzzle_id': puzzle.puzzle_id
                }
                features_list.append(features)
                labels_list.append(1)  # Same group
        
        # Generate negative examples (word pairs in different groups)
        all_words = puzzle.words
        for word1, word2 in combinations(all_words, 2):
            word1_upper = word1.upper()
            word2_upper = word2.upper()
            
            # Check if they're in different groups
            if (word1_upper in word_to_group and 
                word2_upper in word_to_group and
                word_to_group[word1_upper] != word_to_group[word2_upper]):
                
                features = {
                    'word1': word1_upper,
                    'word2': word2_upper,
                    'puzzle_id': puzzle.puzzle_id
                }
                features_list.append(features)
                labels_list.append(0)  # Different groups
    
    return features_list, labels_list


def preprocess_puzzle(puzzle: 'Puzzle') -> 'Puzzle':
    """
    Preprocess a puzzle: clean words and validate structure.
    
    Args:
        puzzle: Raw puzzle
        
    Returns:
        Preprocessed puzzle
    """
    # Import at runtime to avoid circular dependency
    from data.load_dataset import Puzzle
    
    # Clean words
    cleaned_words = clean_puzzle_words(puzzle.words)
    
    # Clean groups if present
    cleaned_groups = {}
    if puzzle.groups:
        for group_id, words in puzzle.groups.items():
            cleaned_groups[group_id] = clean_puzzle_words(words)
    
    # Create new puzzle with cleaned data
    preprocessed = Puzzle(
        puzzle_id=puzzle.puzzle_id,
        words=cleaned_words,
        groups=cleaned_groups if cleaned_groups else puzzle.groups,
        category_descriptions=puzzle.category_descriptions,
        difficulty=puzzle.difficulty,
        contest=puzzle.contest,
        date=puzzle.date
    )
    
    return preprocessed


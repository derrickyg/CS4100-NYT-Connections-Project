"""
Build co-occurrence statistics from historical NYT Connections data.
"""
from collections import defaultdict
from typing import Dict, List, Optional
import math
from data.load_dataset import load_historical_data, Puzzle

# Global cache for co-occurrence stats (loaded once)
_cooccurrence_cache = None


class CooccurrenceStats:
    """Tracks co-occurrence statistics from historical puzzles."""
    
    def __init__(self, exclude_indices: Optional[List[int]] = None):
        """
        Initialize co-occurrence statistics.
        
        Args:
            exclude_indices: Optional list of puzzle indices to exclude from stats
                            (useful for preventing data leakage in testing)
        """
        global _cooccurrence_cache
        
        # Create cache key that includes excluded indices
        cache_key = tuple(sorted(exclude_indices or []))
        
        if _cooccurrence_cache is None:
            _cooccurrence_cache = {}
        
        if cache_key not in _cooccurrence_cache:
            # First time with this exclusion set - build stats
            self.pair_counts = defaultdict(int)  # (word1, word2) -> count
            self.word_counts = defaultdict(int)  # word -> total occurrences
            self.total_puzzles = 0
            self._build_stats(exclude_indices)
            # Cache for future use
            _cooccurrence_cache[cache_key] = {
                'pair_counts': self.pair_counts,
                'word_counts': self.word_counts,
                'total_puzzles': self.total_puzzles
            }
        else:
            # Use cached stats
            cached = _cooccurrence_cache[cache_key]
            self.pair_counts = cached['pair_counts']
            self.word_counts = cached['word_counts']
            self.total_puzzles = cached['total_puzzles']
    
    def _build_stats(self, exclude_indices: Optional[List[int]] = None):
        """Build co-occurrence statistics from historical data."""
        exclude_set = set(exclude_indices or [])
        print("Building co-occurrence statistics from historical puzzles...")
        puzzles = load_historical_data()
        
        # Filter out excluded puzzles
        filtered_puzzles = [p for i, p in enumerate(puzzles) if i not in exclude_set]
        self.total_puzzles = len(filtered_puzzles)
        
        excluded_count = len(puzzles) - len(filtered_puzzles)
        if excluded_count > 0:
            print(f"Excluding {excluded_count} puzzles from stats (to prevent data leakage)...")
        
        print(f"Processing {self.total_puzzles} puzzles...")
        
        for i, puzzle in enumerate(filtered_puzzles):
            if puzzle.groups:
                # Count co-occurrences within groups
                for group_id, words in puzzle.groups.items():
                    for j, word1 in enumerate(words):
                        self.word_counts[word1.upper()] += 1
                        for word2 in words[j+1:]:
                            # Store pairs in sorted order for consistency
                            pair = tuple(sorted([word1.upper(), word2.upper()]))
                            self.pair_counts[pair] += 1
            
            # Progress indicator for large datasets
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{self.total_puzzles} puzzles...")
        
        print(f"✓ Co-occurrence statistics built ({len(self.pair_counts)} unique pairs)")
    
    def cooccurrence_similarity(self, word1: str, word2: str) -> float:
        """
        Compute co-occurrence based similarity using PMI (Pointwise Mutual Information).
        PMI better captures when words co-occur more than expected by chance.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score between 0 and 1
        """
        w1_upper = word1.upper()
        w2_upper = word2.upper()
        
        pair = tuple(sorted([w1_upper, w2_upper]))
        pair_count = self.pair_counts[pair]
        
        if pair_count == 0:
            return 0.0
        
        # Get individual word frequencies
        w1_count = self.word_counts.get(w1_upper, 0)
        w2_count = self.word_counts.get(w2_upper, 0)
        
        if w1_count == 0 or w2_count == 0:
            return 0.0
        
        # Compute PMI: log(P(x,y) / (P(x) * P(y)))
        # P(x,y) = pair_count / total_puzzles
        # P(x) = w1_count / total_puzzles
        # P(y) = w2_count / total_puzzles
        p_xy = pair_count / self.total_puzzles
        p_x = w1_count / self.total_puzzles
        p_y = w2_count / self.total_puzzles
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        pmi = math.log((p_xy + epsilon) / ((p_x * p_y) + epsilon))
        
        # Normalize PMI to 0-1 range
        # PMI can be negative (words co-occur less than expected)
        # PMI can be positive (words co-occur more than expected)
        # Normalize: sigmoid-like function to map to [0, 1]
        # Using tanh normalization: (tanh(pmi) + 1) / 2
        normalized_pmi = (math.tanh(pmi) + 1.0) / 2.0
        
        return normalized_pmi
    
    def get_cooccurrence_count(self, word1: str, word2: str) -> int:
        """Get raw co-occurrence count for a word pair."""
        w1_upper = word1.upper()
        w2_upper = word2.upper()
        pair = tuple(sorted([w1_upper, w2_upper]))
        return self.pair_counts[pair]


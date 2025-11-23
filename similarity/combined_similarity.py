"""
Combine multiple similarity metrics with learned weights.
"""
from typing import Optional, List
from features.word_embeddings import WordEmbeddings
from features.lexical_features import lexical_similarity
from features.cooccurrence import CooccurrenceStats
from features.pattern_matching import pattern_similarity
import config


# Global cache for CombinedSimilarity instance (optional - can be reused)
_combined_similarity_cache = None


class CombinedSimilarity:
    """Combines multiple similarity metrics."""
    
    def __init__(self, use_cache: bool = False, exclude_puzzle_indices: Optional[List[int]] = None):
        """
        Initialize similarity components.
        
        Args:
            use_cache: If True and a cached instance exists, reuse it (default: False for safety)
            exclude_puzzle_indices: Optional list of puzzle indices to exclude from co-occurrence stats
                                   (prevents data leakage when testing on specific puzzles)
        """
        global _combined_similarity_cache
        
        if use_cache and _combined_similarity_cache is not None and exclude_puzzle_indices is None:
            # Reuse cached instance (only if no exclusions)
            cached = _combined_similarity_cache
            self.word_embeddings = cached.word_embeddings
            self.cooccurrence_stats = cached.cooccurrence_stats
        else:
            # Create new instance
            self.word_embeddings = WordEmbeddings(config.EMBEDDING_MODEL)
            self.cooccurrence_stats = CooccurrenceStats(exclude_indices=exclude_puzzle_indices)
            
            # Cache for future use (if requested and no exclusions)
            if use_cache and exclude_puzzle_indices is None:
                _combined_similarity_cache = self
    
    def similarity(self, word1: str, word2: str) -> float:
        """
        Compute combined similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Combined similarity score between 0 and 1
        """
        # Get individual similarity scores
        embedding_sim = self._normalize_similarity(
            self.word_embeddings.cosine_similarity(word1, word2)
        )
        
        cooccurrence_sim = self.cooccurrence_stats.cooccurrence_similarity(word1, word2)
        lexical_sim = lexical_similarity(word1, word2)
        pattern_sim = pattern_similarity(word1, word2)
        
        # Weighted combination
        combined = (
            config.EMBEDDING_WEIGHT * embedding_sim +
            config.COOCCURRENCE_WEIGHT * cooccurrence_sim +
            config.LEXICAL_WEIGHT * lexical_sim +
            config.PATTERN_WEIGHT * pattern_sim
        )
        
        return combined
    
    def _normalize_similarity(self, sim: float) -> float:
        """Normalize similarity from [-1, 1] to [0, 1]."""
        return (sim + 1.0) / 2.0
    
    def average_similarity(self, word: str, group_words: list) -> float:
        """
        Compute average similarity between a word and a group of words.
        
        Args:
            word: Word to compare
            group_words: List of words in the group
            
        Returns:
            Average similarity score
        """
        if len(group_words) == 0:
            return 0.0
        
        total_sim = sum(self.similarity(word, other_word) for other_word in group_words)
        return total_sim / len(group_words)


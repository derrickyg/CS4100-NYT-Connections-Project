"""
Combine multiple similarity metrics with learned weights.
"""
from features.word_embeddings import WordEmbeddings
from features.lexical_features import lexical_similarity
from features.pattern_matching import pattern_similarity
import config


class CombinedSimilarity:
    """Combines multiple similarity metrics."""
    
    def __init__(self):
        """
        Initialize similarity components.
        Always loads models fresh to prevent data leakage.
        """
        # Always create new instance (loads models fresh)
        self.word_embeddings = WordEmbeddings(config.EMBEDDING_MODEL)
    
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
        
        lexical_sim = lexical_similarity(word1, word2)
        pattern_sim = pattern_similarity(word1, word2)
        
        # Weighted combination
        combined = (
            config.EMBEDDING_WEIGHT * embedding_sim +
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


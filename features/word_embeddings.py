"""
Load and use pre-trained word embeddings.
"""
import gensim.downloader as api
from typing import Optional, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
import math

# Global cache for embedding models (loaded once, reused across instances)
_embedding_model_cache = {}

# Global IDF cache for computing TF-IDF weights
_idf_cache = {}


class WordEmbeddings:
    """Wrapper for pre-trained word embeddings with TF-IDF weighting."""
    
    def __init__(self, model_name: str = "glove-wiki-gigaword-300"):
    #def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize word embeddings.
        Uses global cache to avoid reloading models.
        
        Args:
            model_name: Name of the embedding model to load
        """
        self.model_name = model_name
        global _embedding_model_cache
        
        # Check if model is already cached
        if model_name in _embedding_model_cache:
            self.model = _embedding_model_cache[model_name]
            print("HELLOOOOOOOOOO")
            print("New_York" in self.model.key_to_index) 
        else:
            self.model = None
            #self.model = SentenceTransformer(model_name)
            self._load_model()
            # Cache the model for future use
            if self.model is not None:
                _embedding_model_cache[model_name] = self.model
        
        # Initialize IDF cache for this model
        global _idf_cache
        if model_name not in _idf_cache:
            _idf_cache[model_name] = {}
        self.idf_cache = _idf_cache[model_name]
    
    def _load_model(self):
        """Load the embedding model."""
        print(f"Loading word embeddings model: {self.model_name} (this may take 30-60 seconds on first run)...")
        try:
            self.model = api.load(self.model_name)
            #self.model = SentenceTransformer(self.model_name)
            print(f"✓ Successfully loaded {self.model_name}")
        except Exception as e:
            print(f"Warning: Could not load {self.model_name}: {e}")
            print("Falling back to word2vec-google-news-300")
            try:
                self.model = api.load("word2vec-google-news-300")
                print("✓ Successfully loaded word2vec-google-news-300")
            except Exception as e2:
                print(f"Error loading word2vec: {e2}")
                self.model = None
    
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a word or phrase.
        Handles multi-word phrases by using TF-IDF weighted combination of word embeddings.
        
        Args:
            word: Word or phrase to embed
            
        Returns:
            Embedding vector or None if word not found
        """
        if self.model is None:
            return None
        
        # Handle multi-word phrases (e.g., "SOLAR PANEL")
        words = word.split()
        if len(words) > 1:
            # Use TF-IDF weighted combination of individual word embeddings
            return self._get_tfidf_weighted_embedding(words)
        
        # Single word - try different case variations
        return self._get_single_word_embedding(word)
    
    def _get_tfidf_weighted_embedding(self, words: list) -> Optional[np.ndarray]:
        """
        Get TF-IDF weighted combination of word embeddings.
        
        Args:
            words: List of words in the phrase
            
        Returns:
            TF-IDF weighted embedding vector or None if no words found
        """
        # Get embeddings for all words
        embeddings = []
        valid_words = []
        
        for w in words:
            emb = self._get_single_word_embedding(w)
            if emb is not None:
                embeddings.append(emb)
                valid_words.append(w.lower())
        
        if not embeddings:
            return None
        
        # Calculate TF (term frequency) for each word
        word_counts = Counter(valid_words)
        tf = {word: count / len(valid_words) for word, count in word_counts.items()}
        
        # Calculate IDF (inverse document frequency)
        # For simplicity, we use a heuristic: words that appear in fewer positions get higher IDF
        # In a real implementation, this would use a corpus
        idf = {}
        for word in tf.keys():
            if word in self.idf_cache:
                idf[word] = self.idf_cache[word]
            else:
                # Heuristic: assign IDF based on word length and uniqueness
                # Longer, less common words get higher IDF
                base_idf = math.log(1 + len(valid_words) / max(word_counts[word], 1))
                idf[word] = base_idf
                self.idf_cache[word] = base_idf
        
        # Calculate TF-IDF weights and apply them
        weighted_embedding = np.zeros_like(embeddings[0])
        total_weight = 0.0
        
        for word, emb in zip(valid_words, embeddings):
            tfidf_weight = tf[word] * idf[word]
            weighted_embedding += emb * tfidf_weight
            total_weight += tfidf_weight
        
        # Normalize by total weight
        if total_weight > 0:
            weighted_embedding /= total_weight
        
        return weighted_embedding
    
    def _get_single_word_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding for a single word, trying different case variations."""
        word_upper = word.upper()
        word_lower = word.lower()
        word_title = word.title()
        
        # Try different case variations
        for word_variant in [word, word_upper, word_lower, word_title]:
            try:
                return self.model[word_variant]
            except KeyError:
                continue
        
        return None
    
    def cosine_similarity(self, word1: str, word2: str) -> float:
        """
        Compute cosine similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score between -1 and 1, or 0 if words not found
        """
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Normalize vectors (use max to ensure minimum threshold for division)
        emb1_norm = emb1 / max(np.linalg.norm(emb1), 1e-8)
        emb2_norm = emb2 / max(np.linalg.norm(emb2), 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(similarity)


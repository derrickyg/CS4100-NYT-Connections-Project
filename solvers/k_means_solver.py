"""
K-Means clustering approach for grouping 16 words into 4 groups of 4.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from itertools import combinations
import warnings
from features.word_embeddings import WordEmbeddings


class KMeansConnectionsSolver:
    """Solver using K-Means clustering on word embeddings."""
    
    def __init__(self, embeddings_fn: WordEmbeddings):
        """
        Initialize solver with embedding function.
        
        Args:
            embedding_fn: Function that takes a word and returns embedding vector
        """
        self.embeddings_fn = embeddings_fn
    
    def solve_constrained(self, words: List[str], n_init: int = 1000) -> List[List[str]]:
        """
        K-Means with constraint: exactly 4 words per group.
        
        Runs K-Means multiple times and adjusts assignments to ensure
        each cluster has exactly 4 words.
        
        Args:
            words: List of 16 words
            n_init: Number of initializations to try (higher = better, slower)
            
        Returns:
            List of 4 groups, each with exactly 4 words, sorted by cohesion
        """
        if len(words) != 16:
            raise ValueError(f"Expected 16 words, got {len(words)}")
        
        embeddings = np.array([self.embeddings_fn(word) for word in words])
        
        best_groups = None
        best_score = float('inf')
        
        # Try multiple initializations
        for _ in range(n_init):
            # Run K-Means
            kmeans = KMeans(n_clusters=4, n_init=1, max_iter=300)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = kmeans.fit_predict(embeddings)
            
            # Adjust to ensure exactly 4 per group
            groups = self._balance_clusters(
                words, embeddings, labels, kmeans.cluster_centers_
            )
            
            # Score this solution (lower is better)
            score = self._score_groups(groups, embeddings, words)
            
            if score < best_score:
                best_score = score
                best_groups = groups
        
        # Sort by cohesion (tightest groups first)
        best_groups = self._sort_by_cohesion(best_groups, embeddings, words)
        
        return best_groups
    
    def _balance_clusters(self, words: List[str], embeddings: np.ndarray,
                         labels: np.ndarray, centers: np.ndarray) -> List[List[str]]:
        """
        Adjust cluster assignments to ensure exactly 4 words per cluster.
        
        Strategy: Move words between clusters based on distance to centers.
        """
        # Create initial groups
        groups = [[] for _ in range(4)]
        word_to_embedding = {word: emb for word, emb in zip(words, embeddings)}
        
        for word, label in zip(words, labels):
            groups[label].append(word)
        
        # Iteratively balance
        max_iterations = 100
        for _ in range(max_iterations):
            sizes = [len(g) for g in groups]
            
            if all(size == 4 for size in sizes):
                break
            
            # Find largest and smallest clusters
            largest_idx = sizes.index(max(sizes))
            smallest_idx = sizes.index(min(sizes))
            
            if sizes[largest_idx] <= 4:
                break
            
            # Find word in largest cluster closest to smallest cluster center
            best_word = None
            best_distance = float('inf')
            
            for word in groups[largest_idx]:
                emb = word_to_embedding[word]
                dist = np.linalg.norm(emb - centers[smallest_idx])
                if dist < best_distance:
                    best_distance = dist
                    best_word = word
            
            # Move the word
            if best_word:
                groups[largest_idx].remove(best_word)
                groups[smallest_idx].append(best_word)
        
        return groups
    
    def _score_groups(self, groups: List[List[str]], 
                     embeddings: np.ndarray, words: List[str]) -> float:
        """
        Score a grouping solution (lower is better).
        
        Score = sum of within-cluster variance
        """
        word_to_embedding = {word: emb for word, emb in zip(words, embeddings)}
        total_score = 0.0
        
        for group in groups:
            if not group:
                continue
            
            group_embeddings = np.array([word_to_embedding[w] for w in group])
            center = group_embeddings.mean(axis=0)
            
            # Sum of squared distances to center
            variance = np.sum((group_embeddings - center) ** 2)
            total_score += variance
        
        return total_score
    
    def _sort_by_cohesion(self, groups: List[List[str]], 
                         embeddings: np.ndarray, words: List[str]) -> List[List[str]]:
        """Sort groups by internal cohesion (tightest first)."""
        word_to_embedding = {word: emb for word, emb in zip(words, embeddings)}
        
        group_cohesions = []
        for group in groups:
            if not group:
                group_cohesions.append((group, float('inf')))
                continue
            
            group_embeddings = np.array([word_to_embedding[w] for w in group])
            center = group_embeddings.mean(axis=0)
            
            # Average distance to center (lower = more cohesive)
            avg_dist = np.mean([
                np.linalg.norm(emb - center) 
                for emb in group_embeddings
            ])
            group_cohesions.append((group, avg_dist))
        
        # Sort by cohesion (ascending)
        group_cohesions.sort(key=lambda x: x[1])
        
        return [group for group, _ in group_cohesions]
"""
Hierarchical Agglomerative Clustering solver with feedback integration.
Uses bottom-up clustering (merges clusters) instead of partitional (divides).
Re-clusters after each submission using updated similarity weights based on feedback.
"""
import numpy as np
from typing import List, Dict, Optional
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
import warnings
from features.word_embeddings import WordEmbeddings
from evaluation.game_simulator import GameSimulator, GameFeedback


class HierarchicalConnectionsSolver:
    """
    Hierarchical Agglomerative Clustering solver that learns from feedback.
    Re-clusters remaining words after each submission with updated similarity weights.
    Uses bottom-up approach: starts with n clusters, merges until k clusters remain.
    """
    
    def __init__(self, embeddings_fn: WordEmbeddings):
        """
        Initialize solver with embedding function.
        
        Args:
            embeddings_fn: Function that takes a word and returns embedding vector
        """
        self.embeddings_fn = embeddings_fn
        # Weighted similarity matrix: word_pair -- > weight multiplier
        # Starts at 1.0, increases for correct pairs, decreases for wrong pairs
        self.similarity_weights: Dict[tuple, float] = {}
        self.submission_history: List[Dict] = []
    
    def solve_with_feedback(self, game: GameSimulator) -> Dict:
        """
        Solve puzzle using adaptive hierarchical clustering that re-clusters based on feedback.
        
        Args:
            game: GameSimulator instance
            
        Returns:
            Dictionary with solution and statistics
        """
        import time
        start_time = time.time()
        submissions = []
        
        self.similarity_weights.clear()
        self.submission_history.clear()
        
        while not game.is_game_over:
            remaining = game.get_remaining_words()
            if len(remaining) == 0:
                break
            
            # Cluster remaining words with current simil weights
            if len(remaining) <= 4:
                # If 4 or fewer words left, submit those words
                group = remaining
            else:
                # Run adaptive hierarchical clustering on remaining words
                predicted_groups = self._cluster_hierarchical(remaining)
                if not predicted_groups:
                    # Fallback --> just take first 4 words
                    group = remaining[:4]
                else:
                    # Submit the most cohesive group
                    group = predicted_groups[0]
            
            # Submit group
            submission_num = len(submissions) + 1
            feedback = game.submit_group(group)
            submission_data = {
                "group": group,
                "feedback": feedback
            }
            submissions.append(submission_data)
            self.submission_history.append(submission_data)
            
            # Learn from feedback --> update similarity weights
            self._update_similarity_weights(group, feedback)
            
            # Print the correct guesses
            if feedback.is_correct:
                print(f"✓ Correct guess #{submission_num}: {', '.join(group)}")
        
        total_time = time.time() - start_time
        
        # Get final state
        state = game.get_state()
        
        return {
            "solved_groups": game.get_solved_groups(),
            "submissions": submissions,
            "total_submissions": len(submissions),
            "mistakes": state["mistakes"],
            "is_won": state["is_won"],
            "timing": {
                "total": total_time
            }
        }
    
    def _cluster_hierarchical(self, words: List[str]) -> List[List[str]]:
        """
        Cluster words using Hierarchical Agglomerative Clustering with weighted similarity.
        Adapts to feedback through similarity weights.
        
        Args:
            words: List of words to cluster
            
        Returns:
            List of groups, sorted by cohesion (most cohesive first)
        """
        if len(words) < 4:
            return [words] if words else []
        
        # Calc num of groups needed
        num_groups = len(words) // 4
        if num_groups == 0:
            num_groups = 1
        
        # Get embeddings with weighted similarity adjustment
        embeddings = self._get_weighted_embeddings(words)
        
        # Try different linkage methods and pick the best
        best_groups = None
        best_score = float('inf')
        
        linkage_methods = ['ward', 'complete', 'average']
        
        for linkage in linkage_methods:
            try:
                # Run Hierarchical Agglomerative Clustering
                clustering = AgglomerativeClustering(
                    n_clusters=num_groups,
                    linkage=linkage,
                    metric='euclidean'
                )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    labels = clustering.fit_predict(embeddings)
                
                # Convert labels to groups
                groups = [[] for _ in range(num_groups)]
                for word, label in zip(words, labels):
                    if label < num_groups:
                        groups[label].append(word)
                
                # Balance to ensure exactly 4 words per group if possible
                balanced_groups = self._balance_clusters(
                    words, embeddings, labels, groups, num_groups
                )
                
                # Score this solution, lower being better
                score = self._score_groups(balanced_groups, embeddings, words)
                
                if score < best_score:
                    best_score = score
                    best_groups = balanced_groups
            except Exception:
                # If a linkage method fails --> try next one
                continue
        
        if best_groups is None:
            return []
        
        # Sort by cohesion w/ tightest groups being first
        best_groups = self._sort_by_cohesion(best_groups, embeddings, words)
        
        return best_groups
    
    def _get_weighted_embeddings(self, words: List[str]) -> np.ndarray:
        """
        Get embeddings adjusted by similarity weights.
        Words with higher similarity weights are moved closer together in embedding space.
        
        Args:
            words: List of words
            
        Returns:
            Adjusted embedding matrix
        """
        # Get base embeddings
        base_embeddings = np.array([self.embeddings_fn(word) for word in words])
        word_to_idx = {word.upper(): idx for idx, word in enumerate(words)}
        
        # Create adjusted embeddings
        adjusted_embeddings = base_embeddings.copy()
        
        # Apply similarity weights by adjusting embeddings
        # Higher weight --> move embeddings closer together
        for (w1, w2), weight in self.similarity_weights.items():
            if w1 in word_to_idx and w2 in word_to_idx:
                idx1 = word_to_idx[w1]
                idx2 = word_to_idx[w2]
                
                # Calculate current distance
                emb1 = base_embeddings[idx1]
                emb2 = base_embeddings[idx2]
                current_dist = np.linalg.norm(emb1 - emb2)
                
                if current_dist < 1e-8:
                    continue  # Skip if already very close together
                
                # Adjust based on weight
                # Weight > 1.0: move closer
                # Weight < 1.0: move apart
                target_dist = current_dist / weight
                
                # Move embeddings toward/away from each other
                direction = (emb2 - emb1) / (current_dist + 1e-8)
                adjustment = (current_dist - target_dist) * 0.2  # 20% adjustment per iteration
                
                adjusted_embeddings[idx1] += direction * adjustment
                adjusted_embeddings[idx2] -= direction * adjustment
        
        return adjusted_embeddings
    
    def _balance_clusters(self, words: List[str], embeddings: np.ndarray,
                         labels: np.ndarray, groups: List[List[str]], num_groups: int) -> List[List[str]]:
        """
        Adjust cluster assignments to ensure exactly 4 words per cluster when possible.
        
        Args:
            words: List of words
            embeddings: Word embeddings
            labels: Cluster labels
            groups: Initial groups from clustering
            num_groups: Number of groups to create
            
        Returns:
            List of balanced groups
        """
        word_to_embedding = {word: emb for word, emb in zip(words, embeddings)}
        
        # Calculate cluster centers
        centers = []
        for group in groups:
            if group:
                group_embeddings = np.array([word_to_embedding[w] for w in group])
                centers.append(group_embeddings.mean(axis=0))
            else:
                centers.append(np.zeros(embeddings.shape[1]))
        
        # Iteratively balance
        max_iterations = 100
        for _ in range(max_iterations):
            sizes = [len(g) for g in groups]
            
            # Target size per group (4 words each)
            target_size = 4
            
            # Check if we're close to target
            if all(abs(size - target_size) <= 1 for size in sizes):
                break
            
            # Find groups that need adjustment
            oversized = [i for i, size in enumerate(sizes) if size > target_size]
            undersized = [i for i, size in enumerate(sizes) if size < target_size]
            
            if not oversized or not undersized:
                break
            
            # Move word from largest oversized to smallest undersized
            largest_idx = max(oversized, key=lambda i: sizes[i])
            smallest_idx = min(undersized, key=lambda i: sizes[i])
            
            if sizes[largest_idx] <= target_size:
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
                # Update center
                if groups[smallest_idx]:
                    group_embeddings = np.array([word_to_embedding[w] for w in groups[smallest_idx]])
                    centers[smallest_idx] = group_embeddings.mean(axis=0)
        
        # Filter to groups of exactly 4 words
        final_groups = []
        for group in groups:
            if len(group) == 4:
                final_groups.append(group)
            elif len(group) > 4:
                # Take first 4
                final_groups.append(group[:4])
            # Skip groups with < 4 words for now
        
        return final_groups
    
    def _score_groups(self, groups: List[List[str]], 
                     embeddings: np.ndarray, words: List[str]) -> float:
        """
        Score a grouping solution (lower is better).
        
        Score = sum of within-cluster variance
        
        Args:
            groups: List of word groups
            embeddings: Word embeddings
            words: List of all words
            
        Returns:
            Score (lower is better)
        """
        word_to_embedding = {word: emb for word, emb in zip(words, embeddings)}
        total_score = 0.0
        
        for group in groups:
            if len(group) != 4:
                continue
            
            group_embeddings = np.array([word_to_embedding[w] for w in group])
            center = group_embeddings.mean(axis=0)
            
            # Sum of squared distances to center
            variance = np.sum((group_embeddings - center) ** 2)
            total_score += variance
        
        return total_score
    
    def _sort_by_cohesion(self, groups: List[List[str]], 
                         embeddings: np.ndarray, words: List[str]) -> List[List[str]]:
        """
        Sort groups by internal cohesion (tightest first).
        
        Args:
            groups: List of word groups
            embeddings: Word embeddings
            words: List of all words
            
        Returns:
            Sorted groups (most cohesive first)
        """
        word_to_embedding = {word: emb for word, emb in zip(words, embeddings)}
        
        group_cohesions = []
        for group in groups:
            if len(group) != 4:
                continue
            
            group_embeddings = np.array([word_to_embedding[w] for w in group])
            center = group_embeddings.mean(axis=0)
            
            # Average distance to center (lower = more cohesive)
            avg_dist = np.mean([
                np.linalg.norm(emb - center) 
                for emb in group_embeddings
            ])
            group_cohesions.append((group, avg_dist))
        
        # Sort by cohesion w/ ascending, tightest first
        group_cohesions.sort(key=lambda x: x[1])
        
        return [group for group, _ in group_cohesions]
    
    def _update_similarity_weights(self, group: List[str], feedback: GameFeedback):
        """
        Update similarity weights based on feedback.
        This affects future clustering decisions.
        
        Args:
            group: Submitted group of words
            feedback: Feedback received
        """
        group_upper = [w.upper() for w in group]
        
        if feedback.is_correct:
            # All words belong together --> increase similarity weights
            for w1, w2 in combinations(group_upper, 2):
                pair_key = (w1, w2) if w1 < w2 else (w2, w1)
                # Increase weight significantly (multiply by 1.5, cap at 2)
                current = self.similarity_weights.get(pair_key, 1.0)
                self.similarity_weights[pair_key] = min(current * 1.5, 2.0)
        
        elif feedback.correct_words == 0:
            # None of these words belong together --> decrease similarity weights
            for w1, w2 in combinations(group_upper, 2):
                pair_key = (w1, w2) if w1 < w2 else (w2, w1)
                # Decrease weight significantly (multiply by 0.5, floor at 0.1)
                current = self.similarity_weights.get(pair_key, 1.0)
                self.similarity_weights[pair_key] = max(current * 0.5, 0.1)
        
        elif feedback.correct_words >= 2:
            # Partial match --> increase weights for correct words, decrease for wrong
            self._update_partial_match_weights(group_upper, feedback.correct_words)
    
    def _update_partial_match_weights(self, group_upper: List[str], correct_words: int):
        """
        Update similarity weights for partial matches based on similarity scores.
        
        Args:
            group_upper: Words in the group (uppercase)
            correct_words: Number of correct words (2 or 3)
        """
        from similarity.embedding_similarity import EmbeddingSimilarity
        similarity_fn = EmbeddingSimilarity()
        
        # Score all pairs by similarity
        pair_scores = []
        for w1, w2 in combinations(group_upper, 2):
            sim = similarity_fn.similarity(w1, w2)
            pair_scores.append(((w1, w2), sim))
        
        # Sort by similarity
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Number of pairs we expect: C(correct_words, 2)
        num_correct_pairs = correct_words * (correct_words - 1) // 2
        
        # Update weights: top pairs (likely correct) get increased, bottom pairs get decreased
        for idx, ((w1, w2), sim) in enumerate(pair_scores):
            pair_key = (w1, w2) if w1 < w2 else (w2, w1)
            current = self.similarity_weights.get(pair_key, 1.0)
            
            if idx < num_correct_pairs:
                # Top pairs (likely correct): increase weight
                # Weight increases based on similarity (1.2 to 1.6 multiplier)
                multiplier = 1.2 + (sim * 0.4)
                self.similarity_weights[pair_key] = min(current * multiplier, 2.0)
            else:
                # Bottom pairs (likely incorrect): decrease weight
                # Weight decreases based on similarity (0.6 to 0.8 multiplier)
                multiplier = 0.8 - (sim * 0.2)
                self.similarity_weights[pair_key] = max(current * multiplier, 0.1)
        
        # For 3/4 matches, also identify the best trio and strengthen those weights
        if correct_words >= 3:
            best_trio = self._find_best_trio(group_upper)
            if best_trio:
                # Further strengthen weights for the trio
                for w1, w2 in combinations(best_trio, 2):
                    pair_key = (w1, w2) if w1 < w2 else (w2, w1)
                    current = self.similarity_weights.get(pair_key, 1.0)
                    self.similarity_weights[pair_key] = min(current * 1.3, 2.0)
                
                # Further weaken weights involving the 4th word
                bottom_word = [w for w in group_upper if w not in best_trio][0]
                for w in best_trio:
                    pair_key = (bottom_word, w) if bottom_word < w else (w, bottom_word)
                    current = self.similarity_weights.get(pair_key, 1.0)
                    self.similarity_weights[pair_key] = max(current * 0.6, 0.1)
    
    def _find_best_trio(self, group_upper: List[str]) -> Optional[tuple]:
        """Find the 3 words that form the most cohesive group."""
        from similarity.embedding_similarity import EmbeddingSimilarity
        similarity_fn = EmbeddingSimilarity()
        
        best_trio = None
        best_score = float('-inf')
        
        for trio in combinations(group_upper, 3):
            trio_score = sum(
                similarity_fn.similarity(w1, w2)
                for w1, w2 in combinations(trio, 2)
            ) / 3.0  # C(3,2) = 3 pairs
            
            if trio_score > best_score:
                best_score = trio_score
                best_trio = trio
        
        return best_trio


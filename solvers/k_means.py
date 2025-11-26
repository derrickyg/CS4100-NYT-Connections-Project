from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from similarity.combined_similarity import CombinedSimilarity

model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast model

class KMeansSolver:
    """K-Means clustering solver for NYT Connections."""

    def __init__(self, similarity_function: CombinedSimilarity):
        """
        Initialize hill climbing solver.
        
        Args:
            similarity_function: Function to compute word similarity
        """
        self.similarity_fn = similarity_function
    
    def solve(self, words):
        embeddings = model.encode(words)
        
        # One clustering operation - no iterations!
        kmeans = KMeans(n_clusters=4, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Group words in the expected format: Dict[int, List[str]]
        groups = {1: [], 2: [], 3: [], 4: []}
        for j, label in enumerate(labels):
            groups[label + 1].append(words[j])
        
        return groups
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Hyperparameters
embedding_dim = 300
context_size = 2  # Number of context words to use
num_negative_samples = 3  # Number of negative samples per positive sample
learning_rate = 0.001
num_epochs = 5

# Example corpus
# corpus = [
#     "we are what we repeatedly do excellence then is not an act but a habit",
#     "the only way to do great work is to love what you do",
#     "if you can dream it you can do it",
#     "do not wait to strike till the iron is hot but make it hot by striking",
#     "whether you think you can or you think you cannot you are right",
# ]

dataset = load_dataset("tm21cy/NYT-Connections")
word_list = dataset["train"]["words"]
word_list2 = [
  "ARTY", "KISS", "ENAMEL", "ESSAY", "CROWN", "DECAY", "BRUSH", "PASTE", "ANY", "SKIM", "PLASTER", "PULP", "STICK", "STROKE", "ROOT", "FIX"
]
corpus = [item for sub in word_list for item in sub]
corpus = corpus + word_list2

# Preprocess the corpus
def preprocess_corpus(corpus):
    words = [word.strip().upper() for word in corpus]
    vocab = set(words)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return words, word_to_idx, idx_to_word

words, word_to_idx, idx_to_word = preprocess_corpus(corpus)

'''
The function generate_training_data creates training pairs (target, context) 
by considering a window of context words around each target word in the corpus. 
This data will be used to train the Skip-gram model.
'''
# Generate training data
# for nyt we can use the answers to generate context words
# Flatten all answer groups into clusters

# flatten the data into clusters
clusters = []
for puzzle in dataset["train"]["answers"]:        # each puzzle
    for group in puzzle:                           # each answer group in puzzle
        clusters.append(group)                     # group is a dict with "words"

def generate_training_data_clusters(clusters, word_to_idx):
    data = []
    for cluster in clusters:
        words = cluster["words"]
        indices = [word_to_idx[w] for w in words]
        # For each word, all other words in the cluster are context
        for target_idx in indices:
            for context_idx in indices:
                if context_idx != target_idx:  # don't include the word itself
                    data.append((target_idx, context_idx))
    return data


#print(clusters[0]["words"])
training_data = generate_training_data_clusters(clusters, word_to_idx)

'''
A custom PyTorch dataset class,Word2VecDataset, is defined to handle 
the training data. This class is then wrapped in a DataLoader to facilitate 
batching and shuffling during training.
'''
# Custom Dataset class
class Word2VecDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = Word2VecDataset(training_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# clusters: list of dicts with "words"
clusters_indices = []
for cluster in clusters:
    indices = [word_to_idx[w] for w in cluster["words"]]
    clusters_indices.append(indices)

# Build a mapping from target -> its cluster indices
target_to_cluster = {}
for cluster in clusters_indices:
    for idx in cluster:
        target_to_cluster[idx] = cluster

# Negative Sampling
# def get_negative_samples(target, num_negative_samples, vocab_size):
#     neg_samples = []
#     while len(neg_samples) < num_negative_samples:
#         neg_sample = np.random.randint(0, vocab_size)
#         if neg_sample != target:
#             neg_samples.append(neg_sample)
#     return neg_samples

def get_negative_samples(cluster_indices, num_negative_samples, vocab_size):
    """
    Generate negative samples for a target word.
    
    target_idx: index of target word
    cluster_indices: list of word indices in the target's cluster
    num_negative_samples: how many negative samples to generate
    vocab_size: total number of words
    """
    neg_samples = []
    while len(neg_samples) < num_negative_samples:
        neg = np.random.randint(0, vocab_size)
        if neg not in cluster_indices:
            neg_samples.append(neg)
    return neg_samples


# Skip-gram Model with Negative Sampling
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNegSampling, self).__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, target, context, negative_samples):
        target_embedding = self.embeddings(target)
        context_embedding = self.context_embeddings(context)
        negative_embeddings = self.context_embeddings(negative_samples)
        
        positive_score = self.log_sigmoid(torch.sum(target_embedding * context_embedding, dim=1))
        negative_score = self.log_sigmoid(-torch.bmm(negative_embeddings, target_embedding.unsqueeze(2)).squeeze(2)).sum(1)
        
        loss = - (positive_score + negative_score).mean()
        return loss
    
# Training the model
vocab_size = len(word_to_idx)
model = SkipGramNegSampling(vocab_size, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0
    for target, context in dataloader:
        target = target.long()
        context = context.long()

        batch_clusters = [target_to_cluster[t.item()] for t in target]

        negative_samples = torch.LongTensor([get_negative_samples(cluster, num_negative_samples, vocab_size) for cluster in batch_clusters])

        optimizer.zero_grad()
        loss = model(target, context, negative_samples)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


# Getting the word embeddings
embeddings = model.embeddings.weight.detach().numpy()

# Function to get similar words
def get_similar_words(word, top_n=5):
    idx = word_to_idx[word]
    word_embedding = embeddings[idx]
    similarities = np.dot(embeddings, word_embedding)
    closest_idxs = (-similarities).argsort()[1:top_n+1]
    return [idx_to_word[idx] for idx in closest_idxs]


def get_similar_words_subset(word, word_list, top_n=5):
    # Get embeddings for the subset
    subset_indices = [word_to_idx[w] for w in word_list]
    subset_embeddings = embeddings[subset_indices]

    # Get the embedding of the query word
    word_idx = word_to_idx[word]
    word_embedding = embeddings[word_idx]

    # Compute similarities only with the subset
    similarities = np.dot(subset_embeddings, word_embedding)
    
    # Sort and get top indices (skip the word itself if present)
    sorted_idx = (-similarities).argsort()
    top_indices = [i for i in sorted_idx if subset_indices[i] != word_idx][:top_n]

    return [word_list[i] for i in top_indices]

# Example usage
word_list2 = [
  "LASER",
  "PLUCK",
  "THREAD",
  "WAX",
  "COIL",
  "SPOOL",
  "WIND",
  "WRAP",
  "HONEYCOMB",
  "ORGANISM",
  "SOLAR PANEL",
  "SPREADSHEET",
  "BALL",
  "MOVIE",
  "SCHOOL",
  "VITAMIN"
]
print(get_similar_words_subset("LASER", word_list2, top_n=3))


print("Sample training data (target_idx, context_idx):", training_data[:10])

for target_idx, context_idx in training_data[:50]:
    print(f"Target: {idx_to_word[target_idx]}, Context: {idx_to_word[context_idx]}")

#print(len(training_data))
# print(words[:10])
# print("Vocabulary size:", len(word_to_idx))
# print("set size:", len(set(words)))
# print(word_to_idx["LASER"], idx_to_word[1])
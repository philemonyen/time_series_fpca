import numpy as np
from sklearn.manifold import TSNE, trustworthiness

def optimize_tsne_perplexity(score_matrix, n_components=2, k_neighbors=5):
    """
    Finds the optimal perplexity for t-SNE based on the trustworthiness metric.

    Parameters:
    -----------
    score_matrix : array-like, shape (n_samples, n_features)
        The high-dimensional data.
    perplexity_range : list or array
        A sequence of perplexity values to test (e.g., [10, 30, 50, 100]).
    n_components : int, default=2
        Dimension of the embedded space.
    k_neighbors : int, default=5
        Number of neighbors to use when computing trustworthiness.

    Returns:
    --------
    best_perplexity : float
    best_embeddings : numpy.ndarray
    results : dict mapping perplexity to trustworthiness score
    """
    perplexity_range = np.arange(5, int(score_matrix.shape[0] / 20), 5)
    best_score = -1.0
    best_perplexity = None
    best_embeddings = None
    results = {}

    for perp in perplexity_range:
        # 1. Fit t-SNE with the current perplexity
        tsne = TSNE(
            n_components=n_components, 
            perplexity=perp, 
            init="pca", 
            learning_rate="auto",
            random_state=42
        )
        embeddings = tsne.fit_transform(score_matrix)
        
        # 2. Calculate trustworthiness
        # Note: trustworthiness takes the original data and the lower-dimensional embedding
        score = trustworthiness(
            X=score_matrix, 
            X_embedded=embeddings, 
            n_neighbors=k_neighbors
        )
        results[perp] = score

        # 3. Track the best result
        if score > best_score:
            best_score = score
            best_perplexity = perp
            best_embeddings = embeddings
    
    return best_perplexity

def tsne_trasformation(score_matrix, n_components=2):
    """
    Applies t-SNE transformation to the score matrix.
    """
    perplexity = optimize_tsne_perplexity(score_matrix, n_components=n_components)
    tsne = TSNE(
        n_components=n_components, 
        perplexity=perplexity, 
        init="pca", 
        learning_rate="auto", 
        random_state=42)
    return tsne.fit_transform(score_matrix)
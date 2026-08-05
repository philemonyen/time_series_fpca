import numpy as np
import umap
from sklearn.manifold import trustworthiness
import itertools

def tune_umap(X):
    """
    Performs a Grid Search to find the optimal UMAP hyperparameters 
    that maximize local neighborhood trustworthiness.
    
    Parameters:
    X_holdout_fpc   : ndarray - The high-dimensional FPCA scores of the Holdout set.
    n_neighbors_grid: list - Values of n_neighbors to test (e.g., [5, 15, 30, 50]).
    min_dist_grid   : list - Values of min_dist to test (e.g., [0.01, 0.1, 0.5]).
    eval_neighbors  : int - The number of neighbors used by the trustworthiness metric.
    
    Returns:
    best_params : dict - The hyperparameter combination with the highest score.
    results     : list of dicts - The full log of tested parameters and scores.

    trustworthiness should be at least 0.85
    """
    ## Determine mn_neighbors_grid and min_dist_grid
    n_neighbors_grid = [15, 30, 50]
    min_dist_grid = [0, 0.1, 0.5]
    
    best_score = -1.0
    best_reducer = None
    
    # Iterate through every combination of parameters
    for n_neighbors, min_dist in itertools.product(n_neighbors_grid, min_dist_grid):        
        # 1. Fit the UMAP model
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42 # Locked for fair comparison
        )
        
        # Transform the high-dimensional holdout data into 2D
        X_embedded = reducer.fit_transform(X)
        
        # 2. Calculate Trustworthiness
        # Compares the high-dim distances (X) to the low-dim distances (X_embedded)
        score = trustworthiness(
            X, 
            X_embedded, 
            n_neighbors=15, 
            metric='euclidean'
        )
        
        # Update best score
        if score > best_score:
            best_score = score
            best_reducer = reducer    
    return best_reducer
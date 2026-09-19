import numpy as np
from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform, jensenshannon
from scipy.linalg import eigh
from statsmodels.tsa.stattools import acf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cdist
from sklearn.neighbors import NearestNeighbors

### Temporal Metrics
def autocorrelation_score(real_data, synthetic_data, max_lag=24):
    """
    Computes the Autocorrelation Score (ACS) between real and synthetic time series.
    
    Parameters:
    - real_data: np.ndarray of shape (num_samples, seq_len, num_features)
    - synthetic_data: np.ndarray of shape (num_samples, seq_len, num_features)
    - max_lag: int, the maximum time lag to compute the autocorrelation for.
    
    Returns:
    - acs: float, the Mean Absolute Error between the real and synthetic ACF profiles.
           A lower score indicates better preservation of temporal dynamics.
    """
    # Ensure inputs have the same shape
    assert real_data.shape == synthetic_data.shape, "Data shapes must match."
    
    N, T, F = real_data.shape
    
    # Cap max_lag if the sequence length is shorter than the requested lag
    max_lag = min(max_lag, T - 1)
    
    real_acf_profiles = np.zeros((F, max_lag + 1))
    synth_acf_profiles = np.zeros((F, max_lag + 1))
    
    # Iterate through each feature (channel)
    for f in range(F):
        
        # 1. Compute ACF for every sample in the real dataset and average them
        real_acfs = [
            acf(real_data[i, :, f], nlags=max_lag, fft=True) 
            for i in range(N)
        ]
        real_acf_profiles[f] = np.mean(real_acfs, axis=0)
        
        # 2. Compute ACF for every sample in the synthetic dataset and average them
        synth_acfs = [
            acf(synthetic_data[i, :, f], nlags=max_lag, fft=True) 
            for i in range(N)
        ]
        synth_acf_profiles[f] = np.mean(synth_acfs, axis=0)
        
    # 3. Compute the Mean Absolute Error (MAE) across all features and lags
    # We exclude lag 0 because autocorrelation at lag 0 is always exactly 1.0
    acs_score = np.mean(np.abs(real_acf_profiles[:, 1:] - synth_acf_profiles[:, 1:]))
    
    return acs_score

def dtw_score(real_data, synthetic_data, num_samples=100):
    """
    Computes the Expected DTW Distance between real and synthetic time series.
    
    Parameters:
    - real_data: np.ndarray of shape (N, T, F)
    - synthetic_data: np.ndarray of shape (N, T, F)
    - num_samples: int, number of random pairs to evaluate (to avoid O(N^2) bottleneck)
    
    Returns:
    - avg_dtw: float, average DTW distance. Lower is better.
    """
    N_real = len(real_data)
    N_synth = len(synthetic_data)
    
    # Randomly sample indices to create pairs
    idx_real = np.random.choice(N_real, size=num_samples, replace=False)
    idx_synth = np.random.choice(N_synth, size=num_samples, replace=False)
    
    total_dtw = 0.0
    
    for r_idx, s_idx in zip(idx_real, idx_synth):
        # Extract the sequence: shape (T, F)
        seq_real = real_data[r_idx]
        seq_synth = synthetic_data[s_idx]
        
        # fastdtw supports multidimensional sequences automatically
        distance, path = fastdtw(seq_real, seq_synth, dist=euclidean)
        total_dtw += distance
        
    avg_dtw = total_dtw / num_samples
    return avg_dtw

def sample_wise_warping_l2(real_warping_funcs, synth_warping_funcs):
    """
    Calculates the sample-wise temporal fidelity using the L2 norm on warping functions.
    Assumes functions are discretized on identical time grids.
    """
    # Calculate pairwise L2 (Euclidean) distances across the discretized function arrays
    distances = cdist(synth_warping_funcs, real_warping_funcs, metric='euclidean')
    
    # Extract the distance to the nearest real warping function for each synthetic sample
    nearest_neighbor_distances = np.min(distances, axis=1)
    
    # Return the expected temporal precision
    return np.mean(nearest_neighbor_distances)

# Spatial Metrics
def sample_wise_mahalanobis(real_fpc_scores, synth_fpc_scores):
    """
    Calculates the sample-wise morphological fidelity using Mahalanobis distance 
    in the amplitude FPC space.
    """
    # Calculate the covariance matrix of the real FPC scores and its inverse
    # Using pseudoinverse (pinv) prevents crashes if the FPC space is rank-deficient
    cov_matrix = np.cov(real_fpc_scores, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    
    # Calculate pairwise Mahalanobis distances between all synthetic and real samples
    distances = cdist(synth_fpc_scores, real_fpc_scores, metric='mahalanobis', VI=inv_cov_matrix)
    
    # Extract the minimum distance (nearest real neighbor) for each synthetic sample
    nearest_neighbor_distances = np.min(distances, axis=1)
    
    # Return the expected nearest-neighbor precision
    return np.mean(nearest_neighbor_distances)

def wasserstein(X, Y, eps=1e-6):
    """
    Computes the Multidimensional Fréchet Distance (2-Wasserstein distance
    assuming Gaussian distributions) between two matrices.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    mu_X, mu_Y = np.mean(X, axis=0), np.mean(Y, axis=0)
    # np.cov returns a scalar when there is a single feature; keep a 2D matrix.
    sigma_X = np.atleast_2d(np.cov(X, rowvar=False))
    sigma_Y = np.atleast_2d(np.cov(Y, rowvar=False))

    diff = mu_X - mu_Y
    mean_term = diff.dot(diff)

    # SciPy >= 1.16: sqrtm(..., disp=False) is deprecated, and 1x1 inputs skip
    # the (sqrt, errest) tuple and return only the matrix.
    offset = np.eye(sigma_X.shape[0]) * eps
    covmean = sqrtm((sigma_X + offset) @ (sigma_Y + offset))
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    covariance_term = np.trace(sigma_X + sigma_Y - 2 * covmean)
    return float(np.sqrt(max(mean_term + covariance_term, 0.0)))

### UMAP & Diffusion Map Metrics
def grid_js_divergence(real_coords: np.ndarray, 
                                 synthetic_coords: np.ndarray, 
                                 bins: int = 50, 
                                 epsilon: float = 1e-10,
                                 max_dims: int = 2) -> float:
    """
    Computes the Jensen-Shannon Divergence between two sets of coordinates in a shared embedding space
    by constructing an n-dimensional probability density grid.
    
    Parameters:
    -----------
    real_coords : np.ndarray
        Shape (N_real, n_dimensions). E.g., shared 2D UMAP coordinates for real data.
    synthetic_coords : np.ndarray
        Shape (N_synth, n_dimensions). E.g., shared 2D UMAP coordinates for synthetic data.
    bins : int
        Number of grid bins per dimension.
    epsilon : float
        Small smoothing factor to prevent division by zero or log(0).
    max_dims : int
        Number of leading coordinates to histogram. Grid size grows as bins**max_dims, so
        high-dimensional embeddings (e.g. diffusion maps with n_evecs=30) must be truncated.
        
    Returns:
    --------
    float : The JS Divergence bounded between 0.0 (identical) and 1.0 (completely disjoint in log base 2).
    """

    if real_coords.shape[1] > max_dims:
        real_coords = real_coords[:, :max_dims]
        synthetic_coords = synthetic_coords[:, :max_dims]

    # 1. Determine global bounding box across both datasets so grid edges align perfectly
    combined = np.vstack([real_coords, synthetic_coords])
    min_edges = np.min(combined, axis=0)
    max_edges = np.max(combined, axis=0)
    
    # Create bin edges for each dimension
    grid_edges = [np.linspace(min_edges[i], max_edges[i], bins + 1) for i in range(combined.shape[1])]
    
    # 2. Compute N-dimensional histograms
    real_hist, _ = np.histogramdd(real_coords, bins=grid_edges)
    synth_hist, _ = np.histogramdd(synthetic_coords, bins=grid_edges)
    
    # 3. Flatten and apply Laplace/Epsilon smoothing
    real_pdf = real_hist.flatten() + epsilon
    synth_pdf = synth_hist.flatten() + epsilon
    
    # 4. Normalize to valid probability distributions (summing to 1)
    real_pdf /= np.sum(real_pdf)
    synth_pdf /= np.sum(synth_pdf)
    
    # 5. Compute JS Divergence
    # Note: scipy's jensenshannon computes the JS Distance (square root of divergence).
    # We square it and use base=2 so the final divergence is strictly bounded in [0, 1].
    js_distance = jensenshannon(real_pdf, synth_pdf, base=2.0)
    js_divergence = float(js_distance ** 2)
    
    return js_divergence

### Mode Collapse Metrics
def precision_recall(real_features, synthetic_features, k=3):
    """
    Computes manifold-based Precision and Recall for generative models.
    
    Args:
        real_features: 2D numpy array [N_real, feature_dim]
        synthetic_features: 2D numpy array [N_synth, feature_dim]
        k: int, number of nearest neighbors to define the manifold radii
        
    Returns:
        precision: Float (0.0 to 1.0). High precision = good fidelity.
        recall: Float (0.0 to 1.0). High recall = good diversity/coverage.
    """
    
    # 1. Define the Real Manifold Radii
    # We use k+1 because the 1st neighbor of a point is itself (distance 0)
    nn_real = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
    nn_real.fit(real_features)
    distances_real, _ = nn_real.kneighbors(real_features)
    radii_real = distances_real[:, -1] # Distance to the k-th nearest neighbor
    
    # 2. Define the Synthetic Manifold Radii
    nn_synth = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
    nn_synth.fit(synthetic_features)
    distances_synth, _ = nn_synth.kneighbors(synthetic_features)
    radii_synth = distances_synth[:, -1]
    
    # --- PRECISION ---
    # A synthetic point is "precise" if it falls inside the real manifold.
    # We check if the distance to its nearest real point is less than that real point's radius.
    nn_real_1 = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
    nn_real_1.fit(real_features)
    dist_synth_to_real, indices_real = nn_real_1.kneighbors(synthetic_features)
    
    is_in_real_manifold = dist_synth_to_real.squeeze() <= radii_real[indices_real.squeeze()]
    precision = np.mean(is_in_real_manifold)
    
    # --- RECALL ---
    # A real point is "recalled" if it falls inside the synthetic manifold.
    # We check if the distance to its nearest synthetic point is less than that synthetic point's radius.
    nn_synth_1 = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
    nn_synth_1.fit(synthetic_features)
    dist_real_to_synth, indices_synth = nn_synth_1.kneighbors(real_features)
    
    is_in_synth_manifold = dist_real_to_synth.squeeze() <= radii_synth[indices_synth.squeeze()]
    recall = np.mean(is_in_synth_manifold)
    
    return float(precision), float(recall)
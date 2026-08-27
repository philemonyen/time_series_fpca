import numpy as np
from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel
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

# Spatial Metrics
def frechet_score(real_data, synthetic_data, num_samples=100):
    """
    Computes the Expected Fréchet Distance between real and synthetic time series
    using a dynamic programming approach for the discrete Fréchet distance.
    
    Parameters:
    - real_data: np.ndarray of shape (N, T)
    - synthetic_data: np.ndarray of shape (N, T)
    - num_samples: int, number of random pairs to evaluate
    
    Returns:
    - avg_frechet: float, average Fréchet distance. Lower is better.
    """
    N_real = len(real_data)
    N_synth = len(synthetic_data)
    num_samples = min(num_samples, N_real, N_synth)
    
    # Randomly sample indices to create pairs
    idx_real = np.random.choice(N_real, size=num_samples, replace=False)
    idx_synth = np.random.choice(N_synth, size=num_samples, replace=False)
    
    total_frechet = 0.0
    
    for r_idx, s_idx in zip(idx_real, idx_synth):
        # Univariate series as (T, 1) point sequences for cdist
        seq_real = np.asarray(real_data[r_idx], dtype=float).reshape(-1, 1)
        seq_synth = np.asarray(synthetic_data[s_idx], dtype=float).reshape(-1, 1)
        
        # 1. Compute pairwise Euclidean distance matrix between all time steps
        # dist_matrix shape: (T_real, T_synth)
        dist_matrix = cdist(seq_real, seq_synth, metric='euclidean')
        
        T_r, T_s = dist_matrix.shape
        ca = np.zeros((T_r, T_s))
        
        # 2. Dynamic Programming Initialization
        ca[0, 0] = dist_matrix[0, 0]
        
        for i in range(1, T_r):
            ca[i, 0] = max(ca[i-1, 0], dist_matrix[i, 0])
            
        for j in range(1, T_s):
            ca[0, j] = max(ca[0, j-1], dist_matrix[0, j])
            
        # 3. Dynamic Programming Traversal
        for i in range(1, T_r):
            for j in range(1, T_s):
                # The cost is the max of the current spatial distance and the min of the previous path costs
                min_prev_cost = min(
                    ca[i-1, j],    # moving along seq_real
                    ca[i, j-1],    # moving along seq_synth
                    ca[i-1, j-1]   # moving along both
                )
                ca[i, j] = max(dist_matrix[i, j], min_prev_cost)
                
        # The Fréchet distance for this pair is the value at the bottom-right of the cost matrix
        distance = ca[-1, -1]
        total_frechet += distance
        
    avg_frechet = total_frechet / num_samples
    return avg_frechet

def mmd(X, Y, gamma=None):
    """
    Computes the True Maximum Mean Discrepancy using an RBF kernel.
    Captures differences in means, variances, and non-linear shapes.
    """
    # If gamma is None, use the median heuristic or default to 1 / n_features
    if gamma is None:
        gamma = 1.0 / X.shape[1]
        
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    
    # MMD^2 formula
    mmd_squared = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    
    # Relu to prevent tiny negative numbers due to floating point precision
    return np.sqrt(np.max([mmd_squared, 0.0]))

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

def get_diffusion_eigenvalues(data, k=10, sigma=None):
    """
    Constructs the diffusion operator and returns its top k eigenvalues.
    """
    # 1. Compute pairwise squared Euclidean distances
    sq_dists = squareform(pdist(data, metric='sqeuclidean'))
    
    # 2. Estimate kernel bandwidth (sigma) using the median distance heuristic if not provided
    if sigma is None:
        sigma = np.median(sq_dists)
        if sigma == 0.0:
            sigma = 1e-5
            
    # 3. Compute the Gaussian (Heat) affinity matrix W
    W = np.exp(-sq_dists / (2 * sigma))
    
    # 4. Compute the symmetric normalized diffusion matrix
    # D^(-1/2) * W * D^(-1/2) shares the same eigenvalues as the random walk matrix D^(-1) * W
    d = np.sum(W, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    
    # Matrix multiplication for D^(-1/2) * W * D^(-1/2)
    M_sym = W * np.outer(d_inv_sqrt, d_inv_sqrt)
    
    # 5. Extract the top k eigenvalues
    # eigh is optimized for symmetric matrices. It returns eigenvalues in ascending order.
    # We take the last k eigenvalues (which are the largest) and reverse them.
    eigenvalues, _ = eigh(M_sym, subset_by_index=[len(data)-k, len(data)-1])
    top_k_evals = eigenvalues[::-1]
    
    return top_k_evals

def spectral_distance(real_data, synth_data, k=10):
    """
    Computes the Spectral Distance between the diffusion operators of real and synthetic data.
    
    Parameters:
    - real_data: np.ndarray (e.g., FPC score matrix or pre-embedding features)
    - synth_data: np.ndarray 
    - k: int, number of top eigenvalues to compare.
    
    Returns:
    - spectral_dist: float, Mean Squared Error between the top k eigenvalues.
    """
    # Ensure k does not exceed the number of samples
    k = min(k, len(real_data), len(synth_data))
    
    # Calculate eigenvalues
    real_evals = get_diffusion_eigenvalues(real_data, k=k)
    synth_evals = get_diffusion_eigenvalues(synth_data, k=k)
    
    # Compute the distance (MSE) between the eigenvalue spectra
    spectral_dist = np.mean((real_evals - synth_evals) ** 2)
    
    return spectral_dist

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
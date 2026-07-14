import ot
import numpy as np
from typing import Union, Dict
from scipy.linalg import sqrtm
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import cdist, pdist, squareform, jensenshannon
from scipy.spatial import procrustes as scipy_procrustes
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from pydiffmap import diffusion_map as dm

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

def wasserstein(X, Y):
    """
    Computes the Multidimensional Fréchet Distance (2-Wasserstein distance 
    assuming Gaussian distributions) between two matrices.
    """
    mu_X, mu_Y = np.mean(X, axis=0), np.mean(Y, axis=0)
    sigma_X, sigma_Y = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
    
    # Difference between means
    diff = mu_X - mu_Y
    mean_term = diff.dot(diff)
    
    # Product of covariances
    covmean, _ = sqrtm(sigma_X.dot(sigma_Y), disp=False)
    
    # Handle imaginary numbers from numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    covariance_term = np.trace(sigma_X + sigma_Y - 2 * covmean)
    
    return np.sqrt(mean_term + covariance_term)

def gromov_wasserstein(real_scores, synthetic_scores):
    """
    Calculates the Gromov-Wasserstein distance between two independent embeddings.
    
    Parameters:
    real_scores : ndarray, shape (N, D1) - The real holdout data in its Isomap space
    synthetic_scores   : ndarray, shape (M, D2) - The synthetic data in its own Isomap space
    """
    
    # 1. Calculate the intra-space cost matrices (Euclidean distance within each space)
    # C_real represents the geometry of the real manifold
    C_real = cdist(real_scores, real_scores, metric='euclidean')
    
    # C_synth represents the geometry of the synthetic manifold
    C_synth = cdist(synthetic_scores, synthetic_scores, metric='euclidean')
    
    # CRITICAL STEP: Normalization
    # Because Isomap scales can arbitrarily balloon if synthetic data has outliers,
    # you must normalize the distance matrices so they are on the same relative scale.
    C_real /= C_real.max()
    C_synth /= C_synth.max()
    
    # 2. Define the marginal distributions (p and q)
    # We assume every patient/sample has an equal uniform weight of 1/N
    n_real = C_real.shape[0]
    n_synth = C_synth.shape[0]
    p = ot.unif(n_real)
    q = ot.unif(n_synth)
    
    # 3. Compute the Gromov-Wasserstein distance
    # 'square_loss' is the standard loss function for GW
    gw_dist = ot.gromov.gromov_wasserstein2(C_real, C_synth, p, q, 'square_loss')
    
    return gw_dist

def kolmogorov_smirnov(real_scores, synthetic_scores):
    result = []
    for i in range(real_scores.shape[1]):
        stat, pval = ks_2samp(real_scores[:, i], synthetic_scores[:, i])
        result.append(stat)
    return result

def mahalanobis(real_scores: np.ndarray, synthetic_scores: np.ndarray) -> dict:
    """
    Computes the Multivariate Mahalanobis Distance between synthetic scores and the real score distribution.
    
    Parameters:
    -----------
    real_scores : np.ndarray
        Shape (N_real, n_components). The reference scores from real data.
    synthetic_scores : np.ndarray
        Shape (N_synth, n_components). The scores from synthetic data.
        
    Returns:
    --------
    Mean Mahalanobis distance
    """
    # 1. Calculate centroid and covariance of the real scores
    real_mean = np.mean(real_scores, axis=0)
    real_cov = np.cov(real_scores, rowvar=False)
    
    # 2. Use pseudo-inverse (pinv) for numerical stability against singular covariance matrices
    inv_cov = np.linalg.pinv(real_cov)
    
    # 3. Calculate distance for each synthetic sample
    diff = synthetic_scores - real_mean
    
    # Efficient vectorized computation of (x - mu)^T * Sigma^-1 * (x - mu)
    left_term = np.dot(diff, inv_cov)
    mahal_sq = np.sum(left_term * diff, axis=1)
    
    # Ensure no negative values due to floating-point inaccuracies before square root
    mahal_distances = np.sqrt(np.maximum(mahal_sq, 0.0))
    
    return float(np.mean(mahal_distances))

def paired_procrustes(real_coords: np.ndarray, synthetic_coords: np.ndarray) -> Dict[str, float]:
    """
    Computes standard Procrustes Analysis between two PAIRED coordinate matrices of identical shape.
    Use this if your synthetic data has a direct 1-to-1 sample correspondence with the real data
    (e.g., denoising, translation, or comparing ordered structural landmarks).
    
    Parameters:
    -----------
    real_coords : np.ndarray
        Shape (N, d). The reference Isomap coordinates from real data.
    synthetic_coords : np.ndarray
        Shape (N, d). The Isomap coordinates from synthetic data.
        
    Returns:
    --------
    dict containing the Procrustes Disparity score [0, 1] and transformation diagnostics.
    """
    if real_coords.shape != synthetic_coords.shape:
        raise ValueError("Real and Synthetic coordinate matrices must have identical shapes for paired Procrustes.")
        
    # 2. Use the aliased scipy_procrustes here
    mtx1_std, mtx2_aligned, disparity = scipy_procrustes(real_coords, synthetic_coords)
    
    # Extract explicit rotation matrix R using orthogonal procrustes on standardized shapes
    R, _ = orthogonal_procrustes(mtx2_aligned, mtx1_std)
    
    return {
        "procrustes_disparity": float(disparity),
        "similarity_score": float(1.0 - disparity),
        "trace_norm_real": float(np.linalg.norm(mtx1_std, 'fro')),
        "trace_norm_aligned_synth": float(np.linalg.norm(mtx2_aligned, 'fro'))
    }


def unpaired_procrustes(real_coords: np.ndarray, 
                                  synthetic_coords: np.ndarray, 
                                  max_samples: int = 2000) -> Dict[str, float]:
    """
    Computes Procrustes Analysis for UNPAIRED generative datasets by finding the optimal
    topological point correspondence via Linear Sum Assignment (Hungarian Algorithm).
    
    Parameters:
    -----------
    real_coords : np.ndarray
        Shape (N_real, d). Isomap coordinates of real data.
    synthetic_coords : np.ndarray
        Shape (N_synth, d). Isomap coordinates of synthetic data.
    max_samples : int
        Maximum number of samples to use for matching to keep O(N^3) Hungarian algorithm fast.
        
    Returns:
    --------
    dict containing the optimal Procrustes Disparity score after topological pairing.
    """
    # 1. Subsample if datasets are too large to keep matching computationally lightweight
    n_real = len(real_coords)
    n_synth = len(synthetic_coords)
    n_match = min(n_real, n_synth, max_samples)

    real_dim = real_coords.shape[1]
    synth_dim = synthetic_coords.shape[1]
    match_dim = min(real_dim, synth_dim)
    
    idx_real = np.random.choice(n_real, n_match, replace=False) if n_real > n_match else np.arange(n_real)
    idx_synth = np.random.choice(n_synth, n_match, replace=False) if n_synth > n_match else np.arange(n_synth)
    
    X = real_coords[idx_real][:, :match_dim]
    Y = synthetic_coords[idx_synth][:, :match_dim]
    
    # 2. Initial coarse alignment via Principal Component orientation (aligning centroids and axes)
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    X_std = X_centered / np.linalg.norm(X_centered, 'fro')
    Y_std = Y_centered / np.linalg.norm(Y_centered, 'fro')
    
    # SVD alignment of covariance axes
    U_x, _, _ = np.linalg.svd(X_std.T @ X_std)
    U_y, _, _ = np.linalg.svd(Y_std.T @ Y_std)
    Y_pre_aligned = Y_std @ (U_y @ U_x.T)
    
    # 3. Compute pairwise Euclidean distance matrix between pre-aligned clouds
    dist_matrix = cdist(X_std, Y_pre_aligned, metric='euclidean')
    
    # 4. Solve the Linear Sum Assignment (Hungarian algorithm) to find optimal 1-to-1 pairing
    row_indices, col_indices = linear_sum_assignment(dist_matrix)
    
    # 5. Reorder the synthetic coordinates to match the real data's topological order
    Y_matched = Y[col_indices]
    X_matched = X[row_indices]
    
    # 6. Use the aliased scipy_procrustes here to prevent recursion!
    mtx1_std, mtx2_aligned, disparity = scipy_procrustes(X_matched, Y_matched)
    
    return {
        "unpaired_procrustes_disparity": float(disparity),
        "unpaired_similarity_score": float(1.0 - disparity),
        "n_matched_samples": int(n_match)
    }

# FICA dCor
def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Distance Correlation (dCor) between two 1D random vectors.
    """
    x = x.flatten()
    y = y.flatten()
    
    # 1. Compute pairwise Euclidean distance matrices
    a = squareform(pdist(x[:, None], 'euclidean'))
    b = squareform(pdist(y[:, None], 'euclidean'))
    
    # 2. Double centering
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    # 3. Compute distance covariance and variances
    n = x.shape[0]
    dcov2_xy = np.sum(A * B) / (n * n)
    dvar2_x = np.sum(A * A) / (n * n)
    dvar2_y = np.sum(B * B) / (n * n)
    
    if dvar2_x > 0.0 and dvar2_y > 0.0:
        return float(np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dvar2_x) * np.sqrt(dvar2_y)))
    else:
        return 0.0

def dcor_matrix(real_ics: np.ndarray, synthetic_ics: np.ndarray) -> np.ndarray:
    """
    Computes a cross-dCor similarity matrix between Real ICs and Synthetic ICs.
    
    Parameters:
    -----------
    real_ics : np.ndarray
        Shape (N, n_components). The independent components of real data.
    synthetic_ics : np.ndarray
        Shape (N, n_components). The independent components of synthetic data.
        
    Returns:
    --------
    np.ndarray of shape (n_components, n_components) where element [i, j] is dCor(Real_i, Synth_j).
    Ideal fidelity yields an identity-like matrix (high diagonal, zero off-diagonal).
    """
    n_components = real_ics.shape[1]
    dcor_matrix = np.zeros((n_components, n_components))
    
    for i in range(n_components):
        for j in range(n_components):
            dcor_matrix[i, j] = distance_correlation(real_ics[:, i], synthetic_ics[:, j])
            
    return dcor_matrix

# FPCA Principal Component Alignment
def pc_alignment(real_components: np.ndarray, synthetic_components: np.ndarray) -> dict:
    """
    Evaluates the alignment between Real and Synthetic functional principal components/bases.
    
    Parameters:
    -----------
    real_components : np.ndarray
        Shape (n_components, n_features/time_steps). The eigenvectors of the real data.
    synthetic_components : np.ndarray
        Shape (n_components, n_features/time_steps). The eigenvectors of the synthetic data.
        
    Returns:
    --------
    dict containing component-wise absolute cosine similarities and total subspace overlap score [0, 1].
    """
    n_components = real_components.shape[0]
    cosine_similarities = []
    
    for i in range(n_components):
        u = real_components[i]
        v = synthetic_components[i]
        
        # Absolute cosine similarity to neutralize sign-flipping (-1 factor)
        cos_sim = np.abs(np.dot(u.T, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        cosine_similarities.append(float(cos_sim))
        
    # Grassmannian Subspace Overlap (Projection Matrix Similarity)
    # U and V should be orthonormal matrices of shape (features, components)
    U = real_components.squeeze().T
    V = synthetic_components.squeeze().T
    
    # Tr(U^T * V * V^T * U) / k yields a score between 0 (orthogonal) and 1 (identical subspace)
    subspace_overlap = np.trace(U.T @ V @ V.T @ U) / n_components
    
    return {
        "component_wise_cosine_sim": cosine_similarities,
        "mean_cosine_similarity": float(np.mean(cosine_similarities)),
        "subspace_overlap_score": float(subspace_overlap)
    }

# Shared UMAP & Diffusion Map
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

# Diffusion Map
def calculate_von_neumann_entropy(evals: np.ndarray, t: float, epsilon: float = 1e-12) -> float:
    """
    Computes the Von Neumann Entropy of a diffusion process at time step t.
    
    Parameters:
    -----------
    evals : np.ndarray
        1D array of transition matrix eigenvalues.
    t : float
        The diffusion time step / scale.
    epsilon : float
        Small smoothing value to prevent log(0).
        
    Returns:
    --------
    float : The Von Neumann entropy H_VN(t).
    """
    # 1. Power the eigenvalues to simulate diffusion step t
    # Absolute value safeguards against minor numerical precision issues near 0
    powered_evals = np.abs(evals) ** t
    
    # 2. Normalize into a spectral probability distribution p_i(t)
    total_power = np.sum(powered_evals)
    if total_power <= 0.0:
        return 0.0
    
    p_t = powered_evals / total_power
    
    # 3. Filter out zero probabilities for numerical stability in log
    p_t = p_t[p_t > epsilon]
    
    # 4. Compute Shannon entropy of the spectral distribution (Von Neumann entropy)
    # Using natural log (base e); divide by np.log(2) if bits/base-2 are preferred
    entropy = -np.sum(p_t * np.log(p_t))
    
    return float(entropy)


def generate_entropy_curve(dmap_model_or_evals: Union[dm.DiffusionMap, np.ndarray], 
                           t_grid: np.ndarray, 
                           include_trivial: bool = True) -> np.ndarray:
    """
    Generates the Von Neumann entropy curve across a sequence of diffusion time steps.
    
    Parameters:
    -----------
    dmap_model_or_evals : Union[DiffusionMap, np.ndarray]
        A fitted pyDiffMap.DiffusionMap object OR a 1D numpy array of eigenvalues.
    t_grid : np.ndarray
        1D array of diffusion times (e.g., np.logspace(-1, 3, 50)).
    include_trivial : bool
        If True, ensures the stationary eigenvalue lambda_0 = 1.0 is included.
        
    Returns:
    --------
    np.ndarray : The calculated entropy values corresponding to each t in t_grid.
    """
    # 1. Extract eigenvalues from pyDiffMap object if passed directly
    if hasattr(dmap_model_or_evals, 'evals'):
        evals = np.array(dmap_model_or_evals.evals, dtype=float)
    else:
        evals = np.array(dmap_model_or_evals, dtype=float)
        
    # 2. pyDiffMap's .evals drops lambda_0 = 1.0 by default. Prepend it if needed.
    if include_trivial and (len(evals) == 0 or np.max(evals) < 0.9999):
        evals = np.insert(evals, 0, 1.0)
        
    # 3. Calculate entropy for each time step in the grid
    entropy_curve = np.array([calculate_von_neumann_entropy(evals, t) for t in t_grid])
    return entropy_curve

def diffusion_map_entropy_rmse(real_dmap: Union[dm.DiffusionMap, np.ndarray], 
                                     synthetic_dmap: Union[dm.DiffusionMap, np.ndarray], 
                                     t_min: float = 0.1, 
                                     t_max: float = 100.0, 
                                     n_steps: int = 50) -> Dict[str, Union[float, np.ndarray]]:
    """
    Computes the RMSE between the Von Neumann Entropy curves of Real and Synthetic Diffusion Maps.
    
    Parameters:
    -----------
    real_dmap : Union[DiffusionMap, np.ndarray]
        Fitted pyDiffMap object or eigenvalues for the real dataset.
    synthetic_dmap : Union[DiffusionMap, np.ndarray]
        Fitted pyDiffMap object or eigenvalues for the synthetic dataset.
    t_min : float
        Starting diffusion time (short-range local geometry check).
    t_max : float
        Ending diffusion time (long-range global mixing check).
    n_steps : int
        Number of time steps to evaluate along the logarithmic grid.
        
    Returns:
    --------
    dict containing the scalar RMSE, the time grid, and both raw curves for plotting.
    """
    # Create a logarithmically spaced grid of diffusion times to capture both rapid local
    # relaxation (small t) and slow global structural mixing (large t)
    t_grid = np.logspace(np.log10(t_min), np.log10(t_max), n_steps)
    
    # Generate entropy curves
    real_curve = generate_entropy_curve(real_dmap, t_grid)
    synth_curve = generate_entropy_curve(synthetic_dmap, t_grid)
    
    # Compute Root Mean Squared Error across the curve
    squared_errors = (real_curve - synth_curve) ** 2
    rmse = np.sqrt(np.mean(squared_errors))
    
    return {
        "entropy_rmse": float(rmse),
        "t_grid": t_grid,
        "real_entropy_curve": real_curve,
        "synthetic_entropy_curve": synth_curve
    }

# PRDC & LMR
k_list = [3, 5, 10, 30, 50, 100]
def prdc(real_features, fake_features):
    """
    Computes Precision, Recall, Density, and Coverage.
    Args:
        real_features: (N, dim) numpy array of real data embeddings/scores.
        fake_features: (M, dim) numpy array of synthetic data embeddings/scores.
        nearest_k: Number of neighbors to define the manifold clusters.
    """
    n_real = len(real_features)
    n_fake = len(fake_features)

    precisions = []
    recalls = []
    densities = []
    coverages = []
    ks = []

    for k in k_list:
        if k > np.sqrt(n_real) or k > np.sqrt(n_fake):
            break
        ks.append(k)
        
        # 1. Fit Nearest Neighbors on Real Data
        nn_real = NearestNeighbors(n_neighbors=k).fit(real_features)
        dist_real, _ = nn_real.kneighbors(real_features)
        # Radius of the manifold at each real point (distance to k-th neighbor)
        rad_real = dist_real[:, -1]

        # 2. Fit Nearest Neighbors on Fake Data
        nn_fake = NearestNeighbors(n_neighbors=k).fit(fake_features)
        dist_fake, _ = nn_fake.kneighbors(fake_features)
        # Radius of the manifold at each fake point
        rad_fake = dist_fake[:, -1]

        # Distance matrix between all real and all fake points
        # (N, M) matrix: dist_matrix[i, j] is dist(real_i, fake_j)
        dist_real_fake = cdist(real_features, fake_features)

        # --- Precision ---
        # Fraction of fake points that fall into at least one real point's sphere
        # (Checking along columns for fake points)
        precision = np.mean(np.any(dist_real_fake <= rad_real[:, None], axis=0))

        # --- Recall ---
        # Fraction of real points that fall into at least one fake point's sphere
        # (Checking along rows for real points)
        recall = np.mean(np.any(dist_real_fake <= rad_fake[None, :], axis=1))

        # --- Density ---
        # Average number of real spheres that contain a fake point (normalized by k)
        density = np.mean(np.sum(dist_real_fake <= rad_real[:, None], axis=0)) / k

        # --- Coverage ---
        # Fraction of real points that have at least one fake point in their sphere
        coverage = np.mean(np.any(dist_real_fake <= rad_real[:, None], axis=1))

        precisions.append(precision)
        recalls.append(recall)
        densities.append(density)
        coverages.append(coverage)

    return precisions, recalls, densities, coverages, ks

def local_mixing_ratio(real_iso, synth_iso):
    """
    Computes the local mixing ratio of real data in synthetic neighborhoods.
    Args:
        real_iso: (N, dim) numpy array of real data embeddings/scores.
        synth_iso: (M, dim) numpy array of synthetic data embeddings/scores.
    Returns:
        The average ratio of real data in synthetic neighborhoods.
    """
    X_combined = np.vstack((real_iso, synth_iso))
    y_combined = np.concatenate([np.zeros(len(real_iso)), np.ones(len(synth_iso))])

    baseline = len(real_iso) / (len(real_iso) + len(synth_iso))

    # Fit kNN on the joint space
    ratios = [] 
    ks = []
    for k in k_list:
        if k > np.sqrt(len(X_combined)):
            break
        ks.append(k)

        nbrs = NearestNeighbors(n_neighbors=k).fit(X_combined)
        distances, indices = nbrs.kneighbors(X_combined)

        # Isolate synthetic points to check their neighborhoods
        synth_indices = np.where(y_combined == 1)[0]
        neighbor_labels = y_combined[indices[synth_indices, 1:]] # Skip the first index (itself)

        # Calculate ratio of real data in synthetic neighborhoods
        real_ratios = np.mean(neighbor_labels == 0, axis=1)
        ratios.append(np.mean(real_ratios))
    return ratios, baseline, ks
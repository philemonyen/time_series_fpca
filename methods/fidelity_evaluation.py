import ot
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel

#### Distribution-wise Evaluation ####
def mmd_distance(X, Y, gamma=None):
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

def frechet_wasserstein(X, Y):
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

k_list = [3, 5, 10, 30, 50, 100]
def compute_prdc(real_features, fake_features):
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

## Component-wise Evaluation ##
def kolmogorov_smirnov(real_iso, synth_iso):
    result = []
    for i in range(real_iso.shape[1]):
        stat, pval = ks_2samp(real_iso[:, i], synth_iso[:, i])
        result.append(stat)
    return result

## Sample-wise Evaluation ##
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

#### For FPC score matrix similarity
def covariance_operator_dist(X, Y):
    """
    Computes the distance between the covariance operators (matrices) 
    of the FPC scores using the Frobenius Norm.
    """
    # rowvar=False means columns are variables (PCs), rows are samples
    cov_X = np.cov(X, rowvar=False)
    cov_Y = np.cov(Y, rowvar=False)
    
    # Compute the Frobenius norm of the difference matrix
    return np.linalg.norm(cov_X - cov_Y, ord='fro')


def gromov_wasserstein(X_holdout_iso, X_synth_iso):
    """
    Calculates the Gromov-Wasserstein distance between two independent embeddings.
    
    Parameters:
    X_holdout_iso : ndarray, shape (N, D1) - The real holdout data in its Isomap space
    X_synth_iso   : ndarray, shape (M, D2) - The synthetic data in its own Isomap space
    """
    
    # 1. Calculate the intra-space cost matrices (Euclidean distance within each space)
    # C_real represents the geometry of the real manifold
    C_real = cdist(X_holdout_iso, X_holdout_iso, metric='euclidean')
    
    # C_synth represents the geometry of the synthetic manifold
    C_synth = cdist(X_synth_iso, X_synth_iso, metric='euclidean')
    
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

def internal_geometry(X, Y):
    """
    Computes the wasserstein distance between the internal geometries of the two manifolds.
    Compares the distributions of pairwise distances via the upper triangle of each distance matrix.
    """
    X_flat = X[np.triu_indices(X.shape[0], k=1)]
    Y_flat = Y[np.triu_indices(Y.shape[0], k=1)]
    return ot.emd2_1d(X_flat, Y_flat)


# ------ Appendix ------- #
#### Mean Curve Evaluation Metrics ####
# ## Magnitude based
# def mse(fd1, fd2):
#     return np.mean((fd1.data_matrix.squeeze() - fd2.data_matrix.squeeze()) ** 2)

# def rmse(fd1, fd2):
#     return np.sqrt(np.mean((fd1.data_matrix.squeeze() - fd2.data_matrix.squeeze()) ** 2))

# def mae(fd1, fd2):
#     return np.mean(np.abs(fd1.data_matrix.squeeze() - fd2.data_matrix.squeeze()))

# def chebyshev(fd1, fd2):
#     return np.max(np.abs(fd1.data_matrix.squeeze() - fd2.data_matrix.squeeze()))

# ## Correlation-based and Shape-based
# def pearson_correlation(fd1, fd2):
#     return np.corrcoef(fd1.data_matrix.squeeze(), fd2.data_matrix.squeeze())[0, 1]

# def cosine_similarity(fd1, fd2):
#     return np.dot(fd1.data_matrix.squeeze(), fd2.data_matrix.squeeze()) / (np.linalg.norm(fd1.data_matrix.squeeze()) * np.linalg.norm(fd2.data_matrix.squeeze()))

# def coefficient_of_determination(fd1, fd2):
#     return 1 - np.sum((fd1.data_matrix.squeeze() - fd2.data_matrix.squeeze()) ** 2) / np.sum((fd1.data_matrix.squeeze() - np.mean(fd1.data_matrix.squeeze())) ** 2)

# ## Geometric-based
# def frechet_distance(fd1, fd2):
#     return np.linalg.norm(fd1.data_matrix.squeeze() - fd2.data_matrix.squeeze())

# def dtw(fd1, fd2):
#     s1 = fd1.data_matrix.squeeze()
#     s2 = fd2.data_matrix.squeeze()
#     n, m = len(s1), len(s2)
    
#     # Initialize the cost matrix with infinity
#     # We use (n+1) x (m+1) to handle the boundary conditions easily
#     dtw_matrix = np.full((n + 1, m + 1), np.inf)
#     dtw_matrix[0, 0] = 0
    
#     # Fill the matrix
#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             # Euclidean distance between the current points
#             cost = (s1[i-1] - s2[j-1]) ** 2
            
#             # Recurrence relation: add current cost to the minimum of the 3 neighbors
#             dtw_matrix[i, j] = cost + min(
#                 dtw_matrix[i-1, j],    # Insertion
#                 dtw_matrix[i, j-1],    # Deletion
#                 dtw_matrix[i-1, j-1]   # Match
#             )
            
#     # Return the square root of the accumulated cost at the final cell
#     return np.sqrt(dtw_matrix[n, m])

# #### FPC Evaluation Metrics ####
# ## Global Subspace Similarity ##
# def krzanowski_similarity(fd1, fd2, k=None):
#     """
#     Compute the Krzanowski subspace similarity between two sets of eigenfunctions.
#     fd1, fd2: objects with .data_matrix of shape (n_components, n_samples)
#     k: number of leading eigenvectors to use (if None, use min(rank))
#     Returns: Krzanowski similarity score in [0, 1] (1: identical subspaces)
#     """
#     X = fd1.data_matrix.squeeze()
#     Y = fd2.data_matrix.squeeze()

#     # Each row: eigenfunction; so shape (n_eigen, n_samples). We'll treat columns as observations.
#     if X.ndim == 1:
#         X = X[np.newaxis, :]
#     if Y.ndim == 1:
#         Y = Y[np.newaxis, :]

#     r1 = X.shape[0]
#     r2 = Y.shape[0]
#     if k is None:
#         k = min(r1, r2)
#     # QR to orthonormalize leading k eigenvectors
#     Q1, _ = np.linalg.qr(X[:k, :].T)  # shape (n_samples, k)
#     Q2, _ = np.linalg.qr(Y[:k, :].T)  # shape (n_samples, k)

#     # Compute singular values of Q1^T Q2
#     M = np.dot(Q1.T, Q2)
#     s = np.linalg.svd(M, compute_uv=False)
#     similarity = np.sum(s ** 2)  / k # Krzanowski's definition: mean squared singular values

#     return similarity

# def grassmannian_distance(U, V):
#     """
#     Computes Chordal and Geodesic distances between two subspaces.
#     U, V: matrices of shape (n_features, k_components)
#     """
#     # 1. Ensure the bases are orthonormal (essential if FPCs are not pre-normalized)
#     U_orth = orth(U)
#     V_orth = orth(V)
    
#     # 2. Compute the product matrix
#     # If using discretized functional data, ensure this represents the L2 inner product
#     M = U_orth.T @ V_orth
    
#     # 3. Get Singular Values (cosines of the principal angles)
#     # Clip to [0, 1] to avoid numerical errors with arccos
#     S = svd(M, compute_uv=False)
#     S = np.clip(S, 0, 1)
    
#     # 4. Calculate Principal Angles (in radians)
#     angles = np.arccos(S)
    
#     # 5. Geodesic Distance
#     geodesic_dist = np.sqrt(np.sum(angles**2))
    
#     # 6. Chordal Distance
#     # sin^2(theta) = 1 - cos^2(theta)
#     chordal_dist = np.sqrt(np.sum(1 - S**2))
    
#     return geodesic_dist, chordal_dist, np.degrees(angles)
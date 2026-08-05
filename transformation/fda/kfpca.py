import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from skfda.misc.metrics import l2_distance, PairwiseMetric

def get_trapezoidal_weights(grid_points):
    """Helper to compute 1D integration weights."""
    weights = np.zeros_like(grid_points)
    weights[0] = (grid_points[1] - grid_points[0]) / 2.0
    weights[1:-1] = (grid_points[2:] - grid_points[:-2]) / 2.0
    weights[-1] = (grid_points[-1] - grid_points[-2]) / 2.0
    return weights

def functional_gaussian_rbf_kernel(fd_X, gamma):
    """
    Computes the Functional RBF Gaussian kernel using vectorized scikit-learn routines
    and trapezoidal quadrature weights.
    Penalizes sharp spikes heavily
    """
    # 1. Safely extract 2D array without accidentally squeezing a single-sample batch
    X_arr = fd_X.data_matrix.reshape(fd_X.n_samples, -1)
    
    # 2. Extract grid points (assuming 1D domain)
    grid_points = fd_X.grid_points[0]
    
    # 3. Compute trapezoidal integration weights along the domain
    weights = np.zeros_like(grid_points)
    weights[0] = (grid_points[1] - grid_points[0]) / 2.0
    weights[1:-1] = (grid_points[2:] - grid_points[:-2]) / 2.0
    weights[-1] = (grid_points[-1] - grid_points[-2]) / 2.0
    
    # 4. Scale features by sqrt(weights) so Euclidean distance equals L2 integral
    X_scaled = X_arr * np.sqrt(weights)
    
    # 5. Compute squared Euclidean distance matrix directly (highly optimized)
    sq_distance_matrix = euclidean_distances(X_scaled, squared=True)
    
    # 6. Apply the RBF transformation
    K = np.exp(-gamma * sq_distance_matrix)
    
    return K

def functional_laplacian_kernel(fd_X, gamma):
    """
    Laplacian Kernel: Exp(-gamma * L2_norm)
    Much more robust to sharp ECG peaks and phase noise than Gaussian RBF.
    """
    X_arr = fd_X.data_matrix.reshape(fd_X.n_samples, -1)
    weights = get_trapezoidal_weights(fd_X.grid_points[0])
    
    # Scale features by sqrt(weights)
    X_scaled = X_arr * np.sqrt(weights)
    
    # Compute LINEAR Euclidean distance (not squared!)
    distance_matrix = euclidean_distances(X_scaled, squared=False)
    
    # Apply Laplacian decay
    K = np.exp(-gamma * distance_matrix)
    return K

def functional_sobolev_rbf_kernel(fd_X, gamma, lambda_param=1.0):
    """
    Sobolev H1 Kernel: Incorporates both signal amplitude and electrical velocity (slope).
    lambda_param controls how much weight to give to the first derivative.
    """
    # 1. Extract raw amplitudes and first derivatives
    X_arr = fd_X.data_matrix.reshape(fd_X.n_samples, -1)
    
    # Compute 1st derivative using scikit-fda's built-in differentiator
    fd_deriv = fd_X.derivative()
    X_deriv_arr = fd_deriv.data_matrix.reshape(fd_X.n_samples, -1)
    
    # 2. Get integration weights
    weights = get_trapezoidal_weights(fd_X.grid_points[0])
    sqrt_weights = np.sqrt(weights)
    
    # 3. Compute squared L2 distance of amplitudes
    X_scaled = X_arr * sqrt_weights
    sq_dist_amp = euclidean_distances(X_scaled, squared=True)
    
    # 4. Compute squared L2 distance of velocities (derivatives)
    X_deriv_scaled = X_deriv_arr * sqrt_weights
    sq_dist_vel = euclidean_distances(X_deriv_scaled, squared=True)
    
    # 5. Combine into Sobolev H1 squared distance
    sobolev_sq_dist = sq_dist_amp + (lambda_param * sq_dist_vel)
    
    # Apply RBF transformation over the Sobolev space
    K = np.exp(-gamma * sobolev_sq_dist)
    return K

def kfpca_tune_gamma(X):
    """
    Tune gamma by Pre-Image Reconstruction Error Grid Search
    """
    # FIX: Wrap l2_distance in PairwiseMetric and square the resulting distance matrix
    pairwise_sq_dists = PairwiseMetric(l2_distance)(X, X) ** 2
    
    median_sq_dist = np.median(pairwise_sq_dists)
    gamma_init = 1.0 / (2.0 * median_sq_dist)
    gamma_grid = [gamma_init * 2**i for i in range(-3, 4)] # 8 values in total

    best_gamma = None
    best_mse = float('inf')
    for gamma in gamma_grid:
        try:
            K = functional_sobolev_rbf_kernel(X, gamma)
            kpca = KernelPCA(
                n_components=10,
                kernel='precomputed', 
                fit_inverse_transform=True,
                alpha=1.0 # Ridge regression penalty for the inverse transform
            )
            
            embedded_data = kpca.fit_transform(K)
            reconstructed_data = kpca.inverse_transform(embedded_data)
            mse = mean_squared_error(X, reconstructed_data)

            if mse < best_mse:
                best_mse = mse
                best_gamma = gamma
                
        except Exception as e:
            # In rare cases with extreme gammas, the inverse transform matrix can become singular
            print(f"Gamma: {gamma:<6} | Failed to converge ({e})")
    return best_gamma

def kfpca_tuning_n_components(X, gamma, threshold=0.95):
    K = functional_sobolev_rbf_kernel(X, gamma)

    kpca = KernelPCA(n_components=None, kernel='precomputed')
    kpca.fit(K)

    eigenvalues = kpca.eigenvalues_
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    return np.argmax(cumulative_variance >= threshold) + 1

def kfpca_with_param(X, n_components, gamma):
    K = functional_sobolev_rbf_kernel(X, gamma)
    kpca = KernelPCA(n_components=n_components, kernel='precomputed')
    embedding = kpca.fit_transform(K)
    return embedding, kpca
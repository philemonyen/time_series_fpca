import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_squared_error
from skfda.misc.metrics import l2_distance, pairwise_distance

def functional_rbf_kernel(fd_X, gamma):
    """
    Computes the Functional Gaussian (RBF) kernel between two sets of functional data.
    
    The kernel evaluates: K(X, Y) = exp(-gamma * ||X - Y||_{L^2}^2)
    where the L2 norm represents the integral over the continuous domain.

    Parameters:
    -----------
    fd_X : FDataGrid
        The first set of functional data (e.g., training curves).
    gamma : float
        The kernel coefficient. Controls the 'width' of the kernel.

    Returns:
    --------
    K : numpy.ndarray
        The precomputed kernel matrix. 
        Shape is (n_samples_X, n_samples_X)
    """
    # 1. Initialize the functional L2 distance metric
    dist_func = pairwise_distance(l2_distance)
    
    # 2. Compute the integral L2 distances
    distance_matrix = dist_func(fd_X.data_matrix, fd_X.data_matrix)
        
    # 3. Square the distances
    sq_distance_matrix = distance_matrix ** 2
    
    # 4. Apply the RBF/Gaussian transformation
    K = np.exp(-gamma * sq_distance_matrix)
    
    return K

def kfpca_tune_gamma(X):
    """
    Tune gamma by Pre-Image Reconstruction Error Grid Search
    """
    pairwise_sq_dists = pairwise_distance(l2_distance)(X.data_matrix, X.data_matrix)
    median_sq_dist = np.median(pairwise_sq_dists)
    gamma_init = 1.0 / (2.0 * median_sq_dist)
    gamma_grid = [gamma_init * 2**i for i in range(-3, 4)] # 8 values in total


    best_gamma = None
    best_mse = float('inf')
    for gamma in gamma_grid:
        try:
            K = functional_rbf_kernel(X, gamma)
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

def kfpca_tuning_n_components(X, threshold=0.95):
    optimal_gamma = kfpca_tune_gamma(X)
    K = functional_rbf_kernel(X, optimal_gamma)

    kpca = KernelPCA(n_components=None, kernel='precomputed')
    kpca.fit(K)

    eigenvalues = kpca.eigenvalues_
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    return np.argmax(cumulative_variance >= threshold) + 1

def kfpca_with_param(X, n_components, gamma):
    K = functional_rbf_kernel(X, gamma)
    kpca = KernelPCA(n_components=n_components, kernel='precomputed')
    embedding = kpca.fit_transform(K)
    return embedding, kpca
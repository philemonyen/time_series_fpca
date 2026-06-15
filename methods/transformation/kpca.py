import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances

def tune_gamma(X):
    """
    Tune gamma by Pre-Image Reconstruction Error Grid Search
    """
    pairwise_sq_dists = euclidean_distances(X, squared=True)
    median_sq_dist = np.median(pairwise_sq_dists)
    gamma_init = 1.0 / (2.0 * median_sq_dist)
    gamma_grid = [gamma_init * 2**i for i in range(-3, 4)]

    best_gamma = None
    best_mse = float('inf')
    for gamma in gamma_grid:
        try:
            kpca = KernelPCA(
                n_components=10, 
                kernel='rbf', 
                gamma=gamma, 
                fit_inverse_transform=True,
                alpha=1.0 # Ridge regression penalty for the inverse transform
            )
            
            embedded_data = kpca.fit_transform(X)
            reconstructed_data = kpca.inverse_transform(embedded_data)
            mse = mean_squared_error(X, reconstructed_data)

            if mse < best_mse:
                best_mse = mse
                best_gamma = gamma
                
        except Exception as e:
            # In rare cases with extreme gammas, the inverse transform matrix can become singular
            print(f"Gamma: {gamma:<6} | Failed to converge ({e})")
    return best_gamma

def tuning_n_components(X, threshold=0.95):
    optimal_gamma = tune_gamma(X)

    kpca = KernelPCA(n_components=None, kernel='rbf', gamma=optimal_gamma)
    kpca.fit(X)

    eigenvalues = kpca.eigenvalues_
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    return np.argmax(cumulative_variance >= threshold) + 1

def kpca_with_param(X, n_components, gamma):
    kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma)
    embedding = kpca.fit_transform(X)
    return embedding, kpca
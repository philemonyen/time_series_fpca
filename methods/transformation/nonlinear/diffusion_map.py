import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph
from typing import Optional, Dict, Union

class DenseDiffusionMap:
    """
    A crash-proof, dense Diffusion Map implementation with adaptive k-NN bandwidth scaling
    and Nyström out-of-sample extension for generative AI evaluation pipelines.
    """
    def __init__(self, 
                 n_evecs: int = 20, 
                 t: float = 1.0, 
                 k: int = 15, 
                 metric: str = 'cosine', 
                 alpha: float = 1.0,
                 auto_tune_k: bool = True):
        self.n_evecs = n_evecs
        self.t = t
        self.k = k
        self.metric = metric
        self.alpha = alpha  # 1.0 = Laplace-Beltrami (removes sampling density bias)
        self.auto_tune_k = auto_tune_k
        
        # Internal state learned during .fit()
        self.evals_: Optional[np.ndarray] = None
        self.evecs_: Optional[np.ndarray] = None
        self.phi_: Optional[np.ndarray] = None
        self.X_fit_: Optional[np.ndarray] = None
        self.sigma_fit_: Optional[np.ndarray] = None
        self.q_fit_: Optional[np.ndarray] = None
        self.d_fit_: Optional[np.ndarray] = None

    def _ensure_connectivity(self, X: np.ndarray, initial_k: int) -> int:
        """Automatically bumps k if the Markov graph is fragmented into disconnected islands."""
        n_samples = len(X)
        for test_k in [initial_k, 30, 50, 100, 150, 250, 500]:
            if test_k >= n_samples:
                return n_samples - 1
            adj = kneighbors_graph(X, n_neighbors=test_k, mode='connectivity', metric=self.metric)
            n_components, _ = connected_components(adj, directed=False)
            if n_components == 1:
                if test_k > initial_k:
                    print(f"[Auto-Tune] Bumping k from {initial_k} to {test_k} to guarantee graph connectivity.")
                return test_k
        return n_samples - 1

    def fit(self, X: np.ndarray) -> 'DenseDiffusionMap':
        """Fits the Diffusion Map eigendecomposition on reference time-series features."""
        n_samples = len(X)
        self.X_fit_ = np.copy(X)
        
        # 1. Auto-tune k if graph is disconnected
        effective_k = self._ensure_connectivity(X, self.k) if self.auto_tune_k else min(self.k, n_samples - 1)
        
        # 2. Compute pairwise distance matrix using time-series friendly metrics (Cosine/Correlation)
        dist_matrix = squareform(pdist(X, metric=self.metric))
        
        # 3. Adaptive Local Bandwidth (Zelnik-Manor & Perona scaling via k-th nearest neighbor)
        # Sort distances to find the distance to the effective_k-th neighbor for every point
        sorted_dists = np.sort(dist_matrix, axis=1)
        self.sigma_fit_ = np.maximum(sorted_dists[:, effective_k], 1e-12)
        
        # 4. Local Gaussian Kernel: K(x, y) = exp( - d(x,y)^2 / (sigma_x * sigma_y) )
        sigma_outer = np.outer(self.sigma_fit_, self.sigma_fit_)
        K = np.exp(- (dist_matrix ** 2) / sigma_outer)
        np.fill_diagonal(K, 1.0)
        
        # 5. Alpha-normalization (Laplace-Beltrami operator)
        self.q_fit_ = np.maximum(np.sum(K, axis=1), 1e-12)
        q_outer = np.outer(self.q_fit_ ** self.alpha, self.q_fit_ ** self.alpha)
        K_alpha = K / q_outer
        
        # 6. Build Symmetric Markov Transition Matrix for stable eigendecomposition
        self.d_fit_ = np.maximum(np.sqrt(np.sum(K_alpha, axis=1)), 1e-12)
        M_sym = K_alpha / np.outer(self.d_fit_, self.d_fit_)
        
        # 7. DENSE SOLVER: 100% immune to ARPACK "No Convergence" crashes
        evals, evecs = eigh(M_sym)
        
        # Sort descending (largest eigenvalue first)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        # 8. Drop trivial stationary eigenvalue (lambda_0 ~= 1.0) and save valid spectrum
        self.evals_ = evals[1 : self.n_evecs + 1]
        evecs_valid = evecs[:, 1 : self.n_evecs + 1]
        
        # Transform symmetric eigenvectors back to asymmetric Markov coordinates
        self.phi_ = evecs_valid / self.d_fit_[:, None]
        
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits the model and returns the diffusion coordinates scaled by time t."""
        self.fit(X)
        return self.phi_ * (np.abs(self.evals_) ** self.t)

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """
        NYSTRÖM EXTENSION: Projects new (synthetic) time-series data out-of-sample 
        into the reference diffusion space learned during .fit().
        """
        if self.evals_ is None or self.X_fit_ is None:
            raise RuntimeError("You must call .fit() before calling .transform().")
            
        n_new = len(Y)
        
        # 1. Cross-distance matrix between New Data (Y) and Reference Data (X_fit)
        cross_dist = cdist(Y, self.X_fit_, metric=self.metric)
        
        # 2. Estimate local bandwidth for new points based on their k-nearest neighbors in X_fit
        sorted_cross = np.sort(cross_dist, axis=1)
        effective_k = min(self.k, len(self.X_fit_) - 1)
        sigma_new = np.maximum(sorted_cross[:, effective_k], 1e-12)
        
        # 3. Apply Cross-Kernel using reference and new local bandwidths
        sigma_cross = np.outer(sigma_new, self.sigma_fit_)
        K_cross = np.exp(- (cross_dist ** 2) / sigma_cross)
        
        # 4. Alpha-normalization against reference density
        q_new = np.maximum(np.sum(K_cross, axis=1), 1e-12)
        q_cross = np.outer(q_new ** self.alpha, self.q_fit_ ** self.alpha)
        K_alpha_cross = K_cross / q_cross
        
        # 5. Markov transition normalization
        d_new = np.maximum(np.sum(K_alpha_cross, axis=1), 1e-12)
        M_cross = K_alpha_cross / (d_new[:, None] * self.d_fit_[None, :])
        
        # 6. Nyström projection formula: Project onto reference eigenvectors scaled by eigenvalues
        phi_new = (M_cross @ self.phi_) / self.evals_[None, :]
        
        # 7. Scale by diffusion time t to align with reference coordinates
        return phi_new * (np.abs(self.evals_) ** self.t)
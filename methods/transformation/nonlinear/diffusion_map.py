import numpy as np
from pydiffmap.diffusion_map import DiffusionMap

class DynamicDiffusionMap:
    def __init__(self, n_components=2, alpha=1.0, epsilon='bgh'):
        """
        A wrapper around pyDiffMap that allows dynamic time-step (t) adjustments.
        
        Parameters:
        -----------
        n_components : int
            Number of embedded dimensions to compute.
        alpha : float
            Density normalization. 
            alpha=1.0 approximates the Laplace-Beltrami operator (density-invariant).
        epsilon : float or str
            Kernel bandwidth parameter. 
            'bgh' uses the Berry-Giannakis-Harlim method to automatically 
            estimate the optimal bandwidth.
        """
        self.n_components = n_components
        
        # Initialize the pyDiffMap object. 
        # We use from_sklearn to leverage fast nearest-neighbor calculations.
        self.dmap = DiffusionMap.from_sklearn(
            n_evecs=self.n_components, 
            alpha=alpha, 
            epsilon=epsilon
        )

    def fit(self, data_matrix):
        """
        Builds the transition matrix and computes the eigendecomposition.
        This is the computationally expensive step (run once).
        """
        self.dmap.fit(data_matrix)
        return self

    def transform(self, t=1):
        """
        Embeds the data into the lower-dimensional space using time step t.
        This operation is instantaneous.
        
        Parameters:
        -----------
        t : float or int
            The diffusion time step.
            
        Returns:
        --------
        numpy.ndarray, shape (n_samples, n_components)
            The diffusion coordinates at time t.
        """
        if not hasattr(self.dmap, 'evals') or self.dmap.evals is None:
            raise ValueError("You must call fit() before calling transform().")
            
        # pyDiffMap saves the eigenvalues and eigenvectors on the object
        eigenvalues = self.dmap.evals
        eigenvectors = self.dmap.evecs
        
        # Scale each eigenvector by its eigenvalue raised to the power of t
        embedding_t = eigenvectors * (eigenvalues ** t)
        
        return embedding_t

def tune_n_components(X):
    dmap = DiffusionMap.from_sklearn(
        n_evecs=10,
        alpha=1.0,
        epsilon='bgh'
    )
    dmap.fit(X)
    eigenvalues = dmap.evals
    drops = np.abs(np.diff(eigenvalues))
    drop_index = np.argmax(drops)
    return drop_index + 1
import numpy as np
from pydiffmap.diffusion_map import DiffusionMap

def dmap_tune_n_components(X):
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

def dmap_fit(X, n_components):
    dmap = DiffusionMap.from_sklearn(
        n_evecs=n_components,
        alpha=1.0,
        epsilon='bgh'
    )
    dmap.fit(X)
    return dmap
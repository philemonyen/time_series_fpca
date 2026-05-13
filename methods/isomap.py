import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
from kneed import KneeLocator

def find_optimal_k(X, max_k = 50):
    min_k = None

    for k in range(2, max_k + 1):
        adj_matrix = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
        n_comp, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
        
        if n_comp == 1:
            min_k = k
            break

    errors = []
    for k in range(min_k, max_k + 1):
        iso = Isomap(n_neighbors=k, n_components=2)
        iso.fit(X)
        errors.append(iso.reconstruction_error())

    # Find the start of plateau 
    tolerance = 0.01
    deltas = np.abs(np.diff(errors))
    max_drop = np.max(deltas)
    plateau_indices = np.where(deltas < (tolerance * max_drop))[0]
    return plateau_indices[0] + 1

def find_optimal_manifold_dim(X, k):
    errors = []
    for n in range(2, 11):
        iso = Isomap(n_neighbors=k, n_components=n, eigen_solver='dense')
        iso.fit(X)
        errors.append(iso.reconstruction_error())

    # Find the knee point
    kl = KneeLocator(range(2, 11), errors, curve="convex", direction="decreasing", interp_method="polynomial", S=1e-4, online=True)
    return kl.knee
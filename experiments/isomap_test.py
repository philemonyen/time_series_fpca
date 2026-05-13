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
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.fpca import fpca_with_param
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks

save_path = f"../images/isomap/"
path=Path(save_path)
path.mkdir(parents=True, exist_ok=True)

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
    plt.plot(errors)
    plt.xlabel("Number of neighbors")
    plt.ylabel("Reconstruction error")
    plt.title("Reconstruction error vs number of neighbors")
    plt.savefig(save_path + "/reconstruction_error.png")
    plt.close()

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

    plt.plot(errors)
    plt.xlabel("Number of components")
    plt.xlim(2, 10)
    plt.ylabel("Reconstruction error")
    plt.title("Reconstruction error vs number of components")
    plt.savefig(save_path + "/reconstruction_error_manifold_dim.png")
    plt.close()

    # Find the knee point
    kl = KneeLocator(range(2, 11), errors, curve="convex", direction="decreasing", interp_method="polynomial", S=1e-4, online=True)
    return kl.knee

if __name__ == "__main__":
    # Create FPC score matrix
    diagnostic = "NORM"
    lead = 1
    n_data = 1000
    sr = get_sr()
    n_beats = 10
    domain_range = (0, 1)
    n_timepoints = n_beats * sr
    
    # FPCA
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    trimmed_real_fd, landmarks_all = extract_ecg_clinical_landmarks(real_all, n_beats, sr)
    fd = trimmed_real_fd[:n_data]
    landmarks = landmarks_all[:n_data]

    timepoints_per_basis = 2
    n_basis = int(n_timepoints / timepoints_per_basis)
    lambda_ = basis_smoothing_hyperparameter_tuning(fd, n_basis, domain_range)
    fd_smooth, _, _, _ = basis_smoothing_with_lambda(fd, lambda_, n_basis, domain_range)

    fd_aligned, warping_ = landmark_registration(fd_smooth, landmarks)

    n_components = 20
    mean, components, scores, var_ratio, fpca_ = fpca_with_param(fd_aligned, n_components)

    # store score array
    np.save("../data/scores.npy", scores)
    
    # Use the first 10 FPC scores for Isomap
    scores = np.load("../data/scores.npy")
    scores_10 = scores[:, :10]

    # Find find optimal number of neighbors
    optimal_k = find_optimal_k(scores_10)

    # Find optimal number of components i.e., manifold dimension
    optimal_dim = find_optimal_manifold_dim(scores_10, optimal_k)

    print(f"Optimal number of neighbors: {optimal_k}")
    print(f"Optimal number of components: {optimal_dim}")

    iso = Isomap(n_neighbors=optimal_k, n_components=optimal_dim)
    embedding = iso.fit_transform(scores_10)
import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import pickle
import numpy as np
from pathlib import Path
from skfda.representation.grid import FDataGrid
from preprocess.ptbxl_preprocess import (
    get_landmarks, 
    align_ecg, 
)
from transformation.fda.fpca import (
    fpca_with_param, 
    basis_smoothing_hyperparameter_tuning, 
    basis_smoothing_with_lambda, 
    fpca_hyperparameter_tuning,
)
from transformation.nonlinear.diffusion_map import DenseDiffusionMap
from transformation.nonlinear.umap import tune_umap
from metrics.fidelity import *

if __name__ == "__main__":
    results_path = Path("images/sssd_ecg_test")
    results_path.mkdir(parents=True, exist_ok=True)
    result = {}

    data_path_mia = Path("data/validation/privacy/mia/")
    data_path_reid = Path("data/validation/privacy/reidentification/")

    ### Load SSSD-ECG Synthetic Data
    synthetic_ecg = np.load("data/synthetic_final.npy")
    synthetic_label = np.load("data/synthetic_final_labels.npy")
    norm_indices = np.where(synthetic_label[:, 0] == 1)[0]
    synthetic_norm = synthetic_ecg[norm_indices][:, 1, :].squeeze()

    ### Get Real Unaligned and Aligned Data
    with open(f"data/validation/real_data.pkl", "rb") as f:
        real_data = pickle.load(f)
    with open(f"data/validation/real_fd.pkl", "rb") as f:
        real_fd = pickle.load(f)
    n_sample, n_timepoints, n_channel = real_fd.data_matrix.shape
    n_basis = int(n_timepoints / 2)
    with open(f"data/validation/real_segments.pkl", "rb") as f:
        real_segments, real_landmarks = pickle.load(f)

    ### ---- Fidelity Evaluation ---- ###
    ### Morphology & DistributionalFidelity
    # FPCA on real aligned data
    domain_range = (0, 1)
    lambda_ = basis_smoothing_hyperparameter_tuning(real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
    n_components = fpca_hyperparameter_tuning(real_fd_smooth, threshold=0.95)
    # 0.8 -> 31 FPCs, 0.9 -> 44 FPCs, 0.95 -> 74 FPCs
    real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(real_fd_smooth, n_components)
    real_fd_grid = real_fpca_.components_.grid_points[0]
    
    real_score_subset = real_scores[np.random.choice(real_scores.shape[0], size=1500, replace=False)]
    real_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(real_score_subset)
    real_umap = tune_umap(real_score_subset)
    real_dmap_embedding = real_dmap.transform(real_scores)
    real_umap_embedding = real_umap.transform(real_scores)

    # Transform synthetic data to FPCA space
    landmarks = get_landmarks(synthetic_norm, sr=100)
    synthetic_fd = align_ecg(synthetic_norm, landmarks)
    lambda_ = basis_smoothing_hyperparameter_tuning(synthetic_fd, n_basis, domain_range)
    synthetic_fd_smooth, _, _, _ = basis_smoothing_with_lambda(synthetic_fd, lambda_, n_basis, domain_range)
    synthetic_data_matrix = synthetic_fd_smooth(real_fd_grid)
    synthetic_fd_smooth = FDataGrid(data_matrix=synthetic_data_matrix, grid_points=real_fd_grid)

    synthetic_scores = real_fpca_.transform(synthetic_fd_smooth)
    synthetic_dmap = real_dmap.transform(synthetic_scores)
    synthetic_umap = real_umap.transform(synthetic_scores)

    # Metrics
    fpca_wasserstein_score = wasserstein(real_scores, synthetic_scores)
    fpca_mahalanobis_score = sample_wise_mahalanobis(real_scores, synthetic_scores)
    dmap_js_divergence = grid_js_divergence(real_dmap_embedding, synthetic_dmap)
    umap_js_divergence = grid_js_divergence(real_umap_embedding, synthetic_umap)
    precision, recall = precision_recall(real_scores, synthetic_scores)

    # Store results
    result['morphology_wasserstein'] = fpca_wasserstein_score
    result['morphology_mahalanobis'] = fpca_mahalanobis_score
    result['morphology_dmap_js_divergence'] = dmap_js_divergence
    result['morphology_umap_js_divergence'] = umap_js_divergence
    result['morphology_precision'] = precision
    result['morphology_recall'] = recall

    with open(results_path / "sssdecg_morphological_fidelity_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {results_path / 'sssdecg_morphological_fidelity_result.json'}")
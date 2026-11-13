import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import pickle
import numpy as np
from pathlib import Path
from preprocess.ptbxl_preprocess import (
    get_landmarks, 
    extract_ecg_sliding_windows
)
from transformation.fda.fpca import (
    fpca_with_param, 
    basis_smoothing_hyperparameter_tuning, 
    basis_smoothing_with_lambda, 
    fpca_hyperparameter_tuning,
    landmark_registration
)
from transformation.nonlinear.diffusion_map import DenseDiffusionMap
from transformation.nonlinear.umap import tune_umap
from metrics.fidelity import *



if __name__ == "__main__":
    results_path = Path("images/sssd_ecg_test")
    results_path.mkdir(parents=True, exist_ok=True)
    result = {}
    domain_range = (0, 1)

    data_path_mia = Path("data/validation/privacy/mia/")
    data_path_reid = Path("data/validation/privacy/reidentification/")

    ### Load SSSD-ECG Synthetic Data
    synthetic_ecg = np.load("data/synthetic_final.npy")
    synthetic_label = np.load("data/synthetic_final_labels.npy")
    norm_indices = np.where(synthetic_label[:, 0] == 1)[0]
    synthetic_norm = synthetic_ecg[norm_indices][:, 1, :].squeeze()
    synthetic_norm = synthetic_norm[np.random.choice(synthetic_norm.shape[0], size=3000, replace=False)]

    with open(f"data/validation/real_segments.pkl", "rb") as f:
        real_segments, real_landmarks = pickle.load(f)

    ### ---- Fidelity Evaluation ---- ###
    ### Temporal Fidelity
    # Get synthetic aligned segments
    landmarks = get_landmarks(synthetic_norm, sr=100)
    synthetic_segments, synthetic_landmarks = extract_ecg_sliding_windows(synthetic_norm, landmarks)
    aligned_synthetic_segments, synthetic_warping_ = landmark_registration(synthetic_segments, synthetic_landmarks)

    # Apply FPCA on real aligned segments
    aligned_real_segments, real_warping_ = landmark_registration(real_segments, real_landmarks)
    n_basis = int(real_warping_.data_matrix.shape[1] / 2)
    lambda_ = basis_smoothing_hyperparameter_tuning(real_warping_, n_basis, domain_range)
    smoothed_real, _, _, _ = basis_smoothing_with_lambda(real_warping_, lambda_, n_basis, domain_range)
    n_components = fpca_hyperparameter_tuning(smoothed_real)
    real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(smoothed_real, n_components)
    real_score_subset = real_scores[np.random.choice(real_scores.shape[0], size=1000, replace=False)]
    real_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(real_score_subset)
    real_umap = tune_umap(real_score_subset)
    real_dmap_embedding = real_dmap.transform(real_scores)
    real_umap_embedding = real_umap.transform(real_scores)

    # Transform synthetic aligned segments to FPCA space
    synthetic_scores = real_fpca_.transform(synthetic_warping_)
    synthetic_dmap = real_dmap.transform(synthetic_scores)
    synthetic_umap = real_umap.transform(synthetic_scores)

    # Metrics
    fpca_wasserstein_score = wasserstein(real_scores, synthetic_scores)
    fpca_mahalanobis_score = sample_wise_mahalanobis(real_scores, synthetic_scores)
    dmap_js_divergence = grid_js_divergence(real_dmap_embedding, synthetic_dmap)
    umap_js_divergence = grid_js_divergence(real_umap_embedding, synthetic_umap)

    # Store results
    result['temporal_wasserstein'] = fpca_wasserstein_score
    result['temporal_mahalanobis'] = fpca_mahalanobis_score
    result['temporal_dmap_js_divergence'] = dmap_js_divergence
    result['temporal_umap_js_divergence'] = umap_js_divergence

    out_path = results_path / "sssdecg_temporal_fidelity_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {out_path}")
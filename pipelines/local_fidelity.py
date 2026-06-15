import numpy as np
import tabulate as tb
import matplotlib.pyplot as plt
from pathlib import Path
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks, load_synthetic_dataset
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.transformation.fpca import fpca_with_param
from methods.transformation.umap import tune_umap
from methods.evaluation.fidelity import evaluate_meso_fidelity

if __name__ == "__main__":
    diagnostic = "NORM"
    lead = 1
    n_data = 1000
    sr = get_sr()
    n_beats = 10
    domain_range = (0, 1)
    n_timepoints = n_beats * sr
    n_basis = int(n_timepoints / 2)
    n_components = 10
    landmark_locations = np.linspace(0, 1, n_beats+2)[1:-1]

    # Result save path
    save_path = f"images/local_fidelity/"
    path=Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    #### Data Preparation ####
    # Get Real Data
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    trimmed_real_fd, real_landmarks_all = extract_ecg_clinical_landmarks(real_all, n_beats, sr)

    # Get Synthetic Data
    synthetic_all = load_synthetic_dataset(diagnostic, lead)
    trimmed_synthetic_fd, synthetic_landmarks_all = extract_ecg_clinical_landmarks(synthetic_all, n_beats, sr)
    synthetic_fd = trimmed_synthetic_fd[:n_data]
    synthetic_landmarks = synthetic_landmarks_all[:n_data]

    #### Transformation ####
    # Apply FPCA on Real dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(trimmed_real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(trimmed_real_fd, lambda_, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks_all, landmark_locations)
    real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(real_aligned_fd, n_components)

    # Apply Real FPCA on Synthetic
    synthetic_fd_smooth, _, _, _ = basis_smoothing_with_lambda(trimmed_synthetic_fd, lambda_, n_basis, domain_range)
    synthetic_aligned_fd, _ = landmark_registration(synthetic_fd_smooth, synthetic_landmarks_all, landmark_locations)
    synthetic_scores = real_fpca_.transform(synthetic_aligned_fd)

    # Apply UMAP on Real & Synthetic FPCA scores
    real_umap = tune_umap(real_scores)
    real_umap_vec = real_umap.transform(real_scores)
    synthetic_umap_vec = real_umap.transform(synthetic_scores)
    #### ------ Fidelity Evaluation ------ ####
    meso_fidelity = evaluate_meso_fidelity(real_umap_vec, synthetic_umap_vec)
    print(f"Mode Coverage: {meso_fidelity['mode_coverage_ratio']}")
    print(f"JS Divergence Holdout vs Synthetic: {meso_fidelity['js_divergence_holdout_vs_synthetic']}")
    print(f"JS Divergence Real vs Synthetic: {meso_fidelity['js_divergence_real_vs_synthetic']}")
    print(f"Holdout Noise Ratio: {meso_fidelity['holdout_noise_ratio']}")
    print(f"Synthetic Outlier Ratio: {meso_fidelity['synthetic_outlier_ratio']}")

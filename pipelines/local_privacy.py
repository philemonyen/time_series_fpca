import numpy as np
import tabulate as tb
import matplotlib.pyplot as plt
from pathlib import Path
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks, load_synthetic_dataset
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.transformation.fpca import fpca_with_param
from methods.transformation.umap import tune_umap
from methods.evaluation.privacy import domias_vectorized_kde, full_knowledge_mia

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
    save_path = f"images/local_privacy/"
    path=Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    #### Data Preparation ####
    # Get Real Data & Holdout Data
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    trimmed_real_fd, real_landmarks_all = extract_ecg_clinical_landmarks(real_all, n_beats, sr)

    n_samples = trimmed_real_fd.data_matrix.shape[0]
    if n_samples < 2 * n_data:
        raise ValueError(
            f"Need at least {2 * n_data} samples, got {n_samples} after landmark extraction."
        )
    sampled_idx = np.random.choice(n_samples, size=2 * n_data, replace=False)
    real_idx = sampled_idx[:n_data]
    holdout_idx = sampled_idx[n_data:]

    real_fd = trimmed_real_fd[real_idx]
    real_landmarks = real_landmarks_all[real_idx]
    holdout_fd = trimmed_real_fd[holdout_idx]
    holdout_landmarks = real_landmarks_all[holdout_idx]

    # Get Synthetic Data
    synthetic_all = load_synthetic_dataset(diagnostic, lead)
    trimmed_synthetic_fd, synthetic_landmarks_all = extract_ecg_clinical_landmarks(synthetic_all, n_beats, sr)
    synthetic_fd = trimmed_synthetic_fd[:n_data]
    synthetic_landmarks = synthetic_landmarks_all[:n_data]

    #### Transformation ####
    # Apply FPCA on holdout dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(holdout_fd, n_basis, domain_range)
    holdout_fd_smooth, _, _, _ = basis_smoothing_with_lambda(holdout_fd, lambda_, n_basis, domain_range)
    holdout_aligned_fd, _ = landmark_registration(holdout_fd_smooth, holdout_landmarks, landmark_locations)
    holdout_fpca_mean, holdout_fpca_components, holdout_fpca_scores, holdout_fpca_var_ratio, holdout_fpca_ = fpca_with_param(holdout_aligned_fd, n_components)

    # Apply UMAP on holdout FPCA scores
    holdout_umap = tune_umap(holdout_fpca_scores)

    # Apply Holdout FPCA on Real & Synthetic
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks, landmark_locations)
    real_scores = holdout_fpca_.transform(real_aligned_fd)

    synthetic_fd_smooth, _, _, _ = basis_smoothing_with_lambda(synthetic_fd, lambda_, n_basis, domain_range)
    synthetic_aligned_fd, _ = landmark_registration(synthetic_fd_smooth, synthetic_landmarks, landmark_locations)
    synthetic_scores = holdout_fpca_.transform(synthetic_fd)

    # Apply UMAP on Real & Synthetic FPCA scores
    real_umap = holdout_umap.transform(real_scores)
    synthetic_umap = holdout_umap.transform(synthetic_scores)

    #### ------ Privacy Evaluation ------ ####
    umap_density_ratio = domias_vectorized_kde(synthetic_umap, real_umap, holdout_umap)
    avg_umap_privacy = np.mean(umap_density_ratio > 1)
    print(f"Proportion of UMAP DOMIAS Density Ratio > 1: {avg_umap_privacy}")

    # Density Ratio plot #
    log_umap_density_ratio = np.log(umap_density_ratio)
    plt.hist(log_umap_density_ratio, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Log Density Ratio')
    plt.ylabel('Frequency (Count)')
    plt.title('Distribution of Log UMAP Density Ratio')
    plt.savefig(save_path + 'umap_density_ratio.png')

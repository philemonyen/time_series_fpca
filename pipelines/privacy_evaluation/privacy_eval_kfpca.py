import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks, load_synthetic_dataset
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.transformation.fda.kfpca import kfpca_with_param, kfpca_tune_gamma, kfpca_tuning_n_components
from methods.transformation.nonlinear.diffusion_map import dmap_tune_n_components, dmap_fit
from methods.transformation.nonlinear.umap import tune_umap
from methods.transformation.nonlinear.kpca import kpca_tune_n_components, kpca_with_param, tune_gamma
from methods.evaluation.privacy import *

if __name__ == "__main__":
    ## ------------ Data Preparation ------------ ##
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
    save_path = f"images/privacy_eval/kfpca/"
    path=Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

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

    #### ------------ Shared kFPCA ------------ ####
    # Apply kFPCA on Holdout dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(holdout_fd, n_basis, domain_range)
    holdout_fd_smooth, _, _, _ = basis_smoothing_with_lambda(holdout_fd, lambda_, n_basis, domain_range)
    holdout_aligned_fd, _ = landmark_registration(holdout_fd_smooth, holdout_landmarks, landmark_locations)
    kfpca_optimal_gamma = kfpca_tune_gamma(holdout_aligned_fd)
    kfpca_optimal_n_components = kfpca_tuning_n_components(holdout_aligned_fd)
    holdout_kfpca_embedding, holdout_kfpca = kfpca_with_param(holdout_aligned_fd, kfpca_optimal_n_components, kfpca_optimal_gamma)

    # Apply Holdout kFPCA on Real dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks, landmark_locations)
    real_kfpca_embedding = holdout_kfpca.transform(real_aligned_fd)
    
    # Apply Holdout kFPCA on synthetic dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(synthetic_fd, n_basis, domain_range)
    synthetic_fd_smooth, _, _, _ = basis_smoothing_with_lambda(synthetic_fd, lambda_, n_basis, domain_range)
    synthetic_aligned_fd, _ = landmark_registration(synthetic_fd_smooth, synthetic_landmarks_all, landmark_locations)
    synthetic_kfpca_embedding = holdout_kfpca.transform(synthetic_aligned_fd)

    # Evaluation: DOMIAS on kFPCA scores
    kfpca_density_ratio = domias(holdout_kfpca_embedding, real_kfpca_embedding, synthetic_kfpca_embedding)

    # Apply Diffusion Map on holdout kFPCA scores
    holdout_dmap_n_components = dmap_tune_n_components(holdout_kfpca_embedding)
    holdout_dmap = dmap_fit(holdout_kfpca_embedding, holdout_dmap_n_components)
    holdout_dmap_embedding = holdout_dmap.transform(holdout_kfpca_embedding)

    # Apply holdout diffusion map on real kFPCA scores
    real_dmap_embedding = holdout_dmap.transform(real_kfpca_embedding)

    # Apply holdout diffusion map on synthetic kFPCA scores
    synthetic_dmap_embedding = holdout_dmap.transform(synthetic_kfpca_embedding)

    ## Evaluation: DOMIAS on Diffusion map embeddings
    dmap_density_ratio = domias(holdout_dmap_embedding, real_dmap_embedding, synthetic_dmap_embedding)

    # Apply UMAP on holdout kFPCA scores
    holdout_umap = tune_umap(holdout_kfpca_embedding)
    holdout_umap_embedding = holdout_umap.transform(holdout_kfpca_embedding)

    # Apply holdout UMAP on real kFPCA scores
    real_umap_embedding = holdout_umap.transform(real_kfpca_embedding)

    # Apply holdout UMAP on synthetic kFPCA scores
    synthetic_umap_embedding = holdout_umap.transform(synthetic_kfpca_embedding)

    ## Evaluation: DOMIAS on UMAP embeddings
    umap_density_ratio = domias(holdout_umap_embedding, real_umap_embedding, synthetic_umap_embedding)

    # Apply kPCA on holdout kFPCA scores
    holdout_kpca_n_components = kpca_tune_n_components(holdout_kfpca_embedding)
    holdout_kpca_gamma = tune_gamma(holdout_kfpca_embedding)
    holdout_kpca_embedding, holdout_kpca = kpca_with_param(holdout_kfpca_embedding, holdout_kpca_n_components, holdout_kpca_gamma)

    # Apply holdout kPCA on real kFPCA scores
    real_kpca_embedding = holdout_kpca.transform(real_kfpca_embedding)
    # Apply holdout kPCA on synthetic kFPCA scores
    synthetic_kpca_embedding = holdout_kpca.transform(synthetic_kfpca_embedding)
    ## Evaluation: DOMIAS on kPCA embeddings
    kpca_density_ratio = domias(holdout_kpca_embedding, real_kpca_embedding, synthetic_kpca_embedding)

    #### ------------ Result Display ------------ ####
    bandwidth_grid = list(kfpca_density_ratio.keys())
    # kFPCA DOMIAS
    avg_kfpca_privacy = []
    for bandwidth, score in kfpca_density_ratio.items():
        avg= np.mean(score > 0)
        avg_kfpca_privacy.append(avg)

        plt.hist(score, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Log Density Ratio')
        plt.ylabel('Frequency (Count)')
        plt.title(f'Distribution of Log kFPCA Density Ratio (Bandwidth: {bandwidth:.3f})')
        plt.savefig(save_path + f'kfpca_density_ratio_{bandwidth:.3f}.png')
        plt.close()

    plt.plot(bandwidth_grid, avg_kfpca_privacy)
    plt.xlabel('Bandwidth')
    plt.ylabel('Log Density Ratio')
    plt.title('Log kFPCA Density Ratio vs. Kernel Bandwidth')
    plt.savefig(save_path + 'kfpca_density_ratio_vs_bandwidth.png')
    plt.close()

    # Diffusion Map DOMIAS
    bandwidth_grid = list(dmap_density_ratio.keys())
    avg_dmap_privacy = []
    for bandwidth, score in dmap_density_ratio.items():
        avg = np.mean(score > 0)
        avg_dmap_privacy.append(avg)

        plt.hist(score, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Log Density Ratio')
        plt.ylabel('Frequency (Count)')
        plt.title(f'Distribution of Log Diffusion Map Density Ratio (Bandwidth: {bandwidth:.3f})')
        plt.savefig(save_path + f'dmap_density_ratio_{bandwidth:.3f}.png')
        plt.close()
    
    plt.plot(bandwidth_grid, avg_dmap_privacy)
    plt.xlabel('Bandwidth')
    plt.ylabel('Log Density Ratio')
    plt.title('Log Diffusion Map Density Ratio vs. Kernel Bandwidth')
    plt.savefig(save_path + 'dmap_density_ratio_vs_bandwidth.png')
    plt.close()

    # UMAP DOMIAS
    bandwidth_grid = list(umap_density_ratio.keys())
    avg_umap_privacy = []
    for bandwidth, score in umap_density_ratio.items():
        avg = np.mean(score > 0)
        avg_umap_privacy.append(avg)

        plt.hist(score, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Log Density Ratio')
        plt.ylabel('Frequency (Count)')
        plt.title(f'Distribution of Log UMAP Density Ratio (Bandwidth: {bandwidth:.3f})')
        plt.savefig(save_path + f'umap_density_ratio_{bandwidth:.3f}.png')
        plt.close()
    
    plt.plot(bandwidth_grid, avg_umap_privacy)
    plt.xlabel('Bandwidth')
    plt.ylabel('Log Density Ratio')
    plt.title('Log UMAP Density Ratio vs. Kernel Bandwidth')
    plt.savefig(save_path + 'umap_density_ratio_vs_bandwidth.png')
    plt.close()

    # kPCA DOMIAS
    bandwidth_grid = list(kpca_density_ratio.keys())
    avg_kpca_privacy = []
    for bandwidth, score in kpca_density_ratio.items():
        avg = np.mean(score > 0)
        avg_kpca_privacy.append(avg)

        plt.hist(score, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Log Density Ratio')
        plt.ylabel('Frequency (Count)')
        plt.title(f'Distribution of Log kPCA Density Ratio (Bandwidth: {bandwidth:.3f})')
        plt.savefig(save_path + f'kpca_density_ratio_{bandwidth:.3f}.png')
        plt.close()
    
    plt.plot(bandwidth_grid, avg_kpca_privacy)
    plt.xlabel('Bandwidth')
    plt.ylabel('Log Density Ratio')
    plt.title('Log kPCA Density Ratio vs. Kernel Bandwidth')
    plt.savefig(save_path + 'kpca_density_ratio_vs_bandwidth.png')
    plt.close()
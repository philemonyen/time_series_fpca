import sys
from pathlib import Path
# Project root (parent of experiments/) so `methods` resolves when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks, load_synthetic_dataset
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.fpca import fpca_with_param
from methods.isomap import find_optimal_k, find_optimal_manifold_dim
from methods.evaluation import mmd_distance, frechet_wasserstein, covariance_operator_dist, compute_prdc, compute_geometric_score

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
    save_path = f"../images/fidelity/joint_manifold/"
    path=Path(save_path)
    path.mkdir(parents=True, exist_ok=True)

    ### Data Preparation ###
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

    ### FPCA & Isomap on Holdout Data ###
    # Apply FPCA on holdout dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(holdout_fd, n_basis, domain_range)
    holdout_fd_smooth, _, _, _ = basis_smoothing_with_lambda(holdout_fd, lambda_, n_basis, domain_range)
    holdout_aligned_fd, _ = landmark_registration(holdout_fd_smooth, holdout_landmarks, landmark_locations)
    holdout_fpca_mean, holdout_fpca_components, holdout_fpca_scores, holdout_fpca_var_ratio, holdout_fpca_ = fpca_with_param(holdout_aligned_fd, n_components)

    # Apply Isomap on holdout FPC scores
    optimal_k_holdout = find_optimal_k(holdout_fpca_scores)
    optimal_dim_holdout = find_optimal_manifold_dim(holdout_fpca_scores, optimal_k_holdout)
    isomap_holdout = Isomap(n_neighbors=optimal_k_holdout, n_components=optimal_dim_holdout)
    holdout_embedding = isomap_holdout.fit_transform(holdout_fpca_scores)

    ### Apply Holdout FPCA and Isomap on Real & Synthetic ###
    # Apply Holdout FPCA and Isomap on Real
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks, landmark_locations)
    real_scores = holdout_fpca_.transform(real_aligned_fd)
    real_embedding = isomap_holdout.transform(real_scores)

    # Apply Holdout FPCA and Isomap on Synthetic
    synthetic_fd_smooth, _, _, _ = basis_smoothing_with_lambda(synthetic_fd, lambda_, n_basis, domain_range)
    synthetic_aligned_fd, _ = landmark_registration(synthetic_fd_smooth, synthetic_landmarks, landmark_locations)
    synthetic_scores = holdout_fpca_.transform(synthetic_fd)
    synthetic_embedding = isomap_holdout.transform(synthetic_scores)

    ### Fidelity Evaluation ###
    ## Training Gap ##
    # Compare FPC score matrix fidelity
    tg_mmd = mmd_distance(holdout_fpca_scores, real_scores)
    tg_frechet_wasserstein = frechet_wasserstein(holdout_fpca_scores, real_scores)
    tg_covariance_dist = covariance_operator_dist(holdout_fpca_scores, real_scores)

    # Compare Isomap embedding vector fidelity
    tg_precision, tg_recall, tg_density, tg_coverage = compute_prdc(holdout_fpca_scores, real_scores, optimal_k_holdout)
    tg_geometric_score = compute_geometric_score(holdout_fpca_scores, real_scores)

    ## Synthetic Gap ##
    # Compare FPC score matrix fidelity
    sg_mmd = mmd_distance(holdout_fpca_scores, synthetic_scores)
    sg_frechet_wasserstein = frechet_wasserstein(holdout_fpca_scores, synthetic_scores)
    sg_covariance_dist = covariance_operator_dist(holdout_fpca_scores, synthetic_scores)

    # Compare Isomap embedding vector fidelity
    sg_precision, sg_recall, sg_density, sg_coverage = compute_prdc(holdout_fpca_scores, synthetic_scores, optimal_k_holdout)
    sg_geometric_score = compute_geometric_score(holdout_fpca_scores, synthetic_scores)

    ## Real vs. Synthetic ##
    # Compare FPC score matrix fidelity
    rs_mmd = mmd_distance(real_scores, synthetic_scores)
    rs_frechet_wasserstein = frechet_wasserstein(real_scores, synthetic_scores)
    rs_covariance_dist = covariance_operator_dist(real_scores, synthetic_scores)

    # Compare Isomap embedding vector fidelity
    rs_precision, rs_recall, rs_density, rs_coverage = compute_prdc(real_scores, synthetic_scores, optimal_k_holdout)
    rs_geometric_score = compute_geometric_score(real_scores, synthetic_scores)
    
    #### Result Display & Save ####
    # Holdout FPCA Plots
    holdout_fd.plot()
    plt.title("Holdout Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig(save_path + "/holdout_data.png")
    plt.close()
    holdout_fd_smooth.plot()
    plt.title("Holdout Data (Smoothed)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig(save_path + "/holdout_data_smoothed.png")
    plt.close()
    holdout_aligned_fd.plot()
    plt.title("Holdout Data (Aligned)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig(save_path + "/holdout_data_aligned.png")
    plt.close()
    holdout_fpca_mean.plot()
    plt.title("Holdout Data (FPCA Mean)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig(save_path + "/holdout_data_fpca_mean.png")
    plt.close()
    for i in range(n_components):
        plt.plot(holdout_fpca_components[i].data_matrix.squeeze())
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"Holdout FPCA Component {i+1}")
        plt.savefig(save_path + f"/holdout_fpca_component_{i+1}.png")
        plt.close()

    # Training Gap Isomap Pairwise Plot
    holdout_labels = np.array(["Holdout"] * holdout_embedding.shape[0]).reshape(-1, 1)
    real_labels = np.array(["Real"] * real_embedding.shape[0]).reshape(-1, 1)
    embeddings = np.concatenate((holdout_embedding[:, :optimal_dim_holdout], real_embedding[:, :optimal_dim_holdout]), axis=0)
    labels = np.concatenate((holdout_labels, real_labels), axis=0)
    df = pd.DataFrame(embeddings, columns=[f'Component {i+1}' for i in range(optimal_dim_holdout)])
    df['Source'] = labels
    sns.pairplot(df, hue='Source', palette='viridis', diag_kind='kde', plot_kws={'alpha': 0.2})
    plt.suptitle('Isomap Component Matrix: Holdout vs Real', y=1.02)
    plt.legend(labels=['Holdout', 'Real'], loc='upper right')
    plt.savefig(save_path + "/traing_gap_isomap_pairwise.png")
    plt.close()

    # Synthetic Gap Isomap Pairwise Plot
    synthetic_labels = np.array(["Synthetic"] * synthetic_embedding.shape[0]).reshape(-1, 1)
    embeddings = np.concatenate((holdout_embedding[:, :optimal_dim_holdout], synthetic_embedding[:, :optimal_dim_holdout]), axis=0)
    labels = np.concatenate((holdout_labels, synthetic_labels), axis=0)
    df = pd.DataFrame(embeddings, columns=[f'Component {i+1}' for i in range(optimal_dim_holdout)])
    df['Source'] = labels
    sns.pairplot(df, hue='Source', palette='viridis', diag_kind='kde', plot_kws={'alpha': 0.2})
    plt.suptitle('Isomap Component Matrix: Holdout vs Synthetic', y=1.02)
    plt.legend(labels=['Holdout', 'Synthetic'], loc='upper right')
    plt.savefig(save_path + "/synthetic_gap_isomap_pairwise.png")
    plt.close()

    # Real vs. Synthetic Isomap Pairwise Plot
    embeddings = np.concatenate((real_embedding[:, :optimal_dim_holdout], synthetic_embedding[:, :optimal_dim_holdout]), axis=0)
    labels = np.concatenate((real_labels, synthetic_labels), axis=0)
    df = pd.DataFrame(embeddings, columns=[f'Component {i+1}' for i in range(optimal_dim_holdout)])
    df['Source'] = labels
    sns.pairplot(df, hue='Source', palette='viridis', diag_kind='kde', plot_kws={'alpha': 0.2})
    plt.suptitle('Isomap Component Matrix: Real vs Synthetic', y=1.02)
    plt.legend(labels=['Real', 'Synthetic'], loc='upper right')
    plt.savefig(save_path + "/real_vs_synthetic_isomap_pairwise.png")
    plt.close()

    # Hyperparameter Setting
    print(f"Smoothing Parameter: {lambda_}")
    print(f"Number of FPCs: {n_components}")
    print(f"Varaince Ratio Sum: {np.sum(holdout_fpca_var_ratio)}")
    print(f"Optimal Neighborhood Range (# neighbors): {optimal_k_holdout}")
    print(f"Optimal Manifold Dimension: {optimal_dim_holdout}")

    # FPC score matrix fidelity
    print(f"Training Gap MMD distance: {tg_mmd}")
    print(f"Training Gap Frechet Wasserstein distance: {tg_frechet_wasserstein}")
    print(f"Training Gap Covariance operator distance: {tg_covariance_dist}")
    print(f"Synthetic Gap MMD distance: {sg_mmd}")
    print(f"Synthetic Gap Frechet Wasserstein distance: {sg_frechet_wasserstein}")
    print(f"Synthetic Gap Covariance operator distance: {sg_covariance_dist}")

    # Isomap embedding vector fidelity
    print(f"Training Gap Precision: {tg_precision}")
    print(f"Training Gap Recall: {tg_recall}")
    print(f"Training Gap Density: {tg_density}")
    print(f"Training Gap Coverage: {tg_coverage}")
    print(f"Training Gap Geometric score: {tg_geometric_score}")
    print(f"Synthetic Gap Precision: {sg_precision}")
    print(f"Synthetic Gap Recall: {sg_recall}")
    print(f"Synthetic Gap Density: {sg_density}")
    print(f"Synthetic Gap Coverage: {sg_coverage}")
    print(f"Synthetic Gap Geometric score: {sg_geometric_score}")

    # Real vs. Synthetic FPC score matrix fidelity
    print(f"Real vs. Synthetic MMD distance: {rs_mmd}")
    print(f"Real vs. Synthetic Covariance operator distance: {rs_covariance_dist}")
    print(f"Real vs. Synthetic Frechet Wasserstein distance: {rs_frechet_wasserstein}")

    # Real vs. Synthetic Isomap embedding vector fidelity
    print(f"Real vs. Synthetic Precision: {rs_precision}")
    print(f"Real vs. Synthetic Recall: {rs_recall}")
    print(f"Real vs. Synthetic Density: {rs_density}")
    print(f"Real vs. Synthetic Coverage: {rs_coverage}")
    print(f"Real vs. Synthetic Geometric score: {rs_geometric_score}")
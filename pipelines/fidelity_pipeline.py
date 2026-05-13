import sys
from pathlib import Path
# Project root (parent of experiments/) so `methods` resolves when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks, load_synthetic_dataset
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.fpca import fpca_with_param
from methods.isomap import find_optimal_k, find_optimal_manifold_dim
from methods.evaluation import euclidean, krzanowski_similarity

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

    # Result save path
    save_path = f"../images/fidelity/"

    # Get Real Data
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    trimmed_real_fd, real_landmarks_all = extract_ecg_clinical_landmarks(real_all, n_beats, sr)
    real_fd = trimmed_real_fd[:n_data]
    real_landmarks = real_landmarks_all[:n_data]

    # Get Synthetic Data
    synthetic_all = load_synthetic_dataset(diagnostic, lead)
    trimmed_synthetic_fd, synthetic_landmarks_all = extract_ecg_clinical_landmarks(synthetic_all, n_beats, sr)
    synthetic_fd = trimmed_synthetic_fd[:n_data]
    synthetic_landmarks = synthetic_landmarks_all[:n_data]

    # Apply FPCA on Real and Synthetic
    lambda_ = basis_smoothing_hyperparameter_tuning(real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks)
    lambda_ = basis_smoothing_hyperparameter_tuning(synthetic_fd, n_basis, domain_range)
    synthetic_fd_smooth, _, _, _ = basis_smoothing_with_lambda(synthetic_fd, lambda_, n_basis, domain_range)
    synthetic_aligned_fd, _ = landmark_registration(synthetic_fd_smooth, synthetic_landmarks)

    # Apply FPCA on Real and Synthetic - extract the first 10 FPC
    real_fpca_mean, real_fpca_components, real_fpca_scores, real_fpca_var_ratio, real_fpca_ = fpca_with_param(real_aligned_fd, n_components)
    synthetic_fpca_mean, synthetic_fpca_components, synthetic_fpca_scores, synthetic_fpca_var_ratio, synthetic_fpca_ = fpca_with_param(synthetic_aligned_fd, n_components)
    print(f"Variance ratio sum - Real: {np.sum(real_fpca_var_ratio)}")
    print(f"Variance ratio sum - Synthetic: {np.sum(synthetic_fpca_var_ratio)}")

    # Isomap on Real and Synthetic FPC scores
    real_optimal_k = find_optimal_k(real_fpca_scores)
    real_optimal_dim = find_optimal_manifold_dim(real_fpca_scores, real_optimal_k)
    synthetic_optimal_k = find_optimal_k(synthetic_fpca_scores)
    synthetic_optimal_dim = find_optimal_manifold_dim(synthetic_fpca_scores, synthetic_optimal_k)
    print(f"Optimal number of neighbors - Real: {real_optimal_k}")
    print(f"Optimal number of components - Real: {real_optimal_dim}")
    print(f"Optimal number of neighbors - Synthetic: {synthetic_optimal_k}")
    print(f"Optimal number of components - Synthetic: {synthetic_optimal_dim}")

    real_iso = Isomap(n_neighbors=real_optimal_k, n_components=real_optimal_dim)
    synthetic_iso = Isomap(n_neighbors=synthetic_optimal_k, n_components=synthetic_optimal_dim)
    real_embedding = real_iso.fit_transform(real_fpca_scores)
    synthetic_embedding = synthetic_iso.fit_transform(synthetic_fpca_scores)
    
    # Plot
    shared_n = min(real_optimal_dim, synthetic_optimal_dim)
    real_labels = np.array(["Real"] * real_embedding.shape[0]).reshape(-1, 1)
    synthetic_labels = np.array(["Synthetic"] * synthetic_embedding.shape[0]).reshape(-1, 1)
    embeddings = np.concatenate((real_embedding[:, :shared_n], synthetic_embedding[:, :shared_n]), axis=0)
    labels = np.concatenate((real_labels, synthetic_labels), axis=0)
    df = pd.DataFrame(embeddings, columns=[f'Component {i+1}' for i in range(real_optimal_dim)])
    df['Source'] = labels

    sns.pairplot(df, hue='Source', palette='viridis', diag_kind='kde', plot_kws={'alpha': 0.2})
    plt.suptitle('Isomap Component Matrix: Real NORM vs Synthetic NORM', y=1.02)
    plt.legend(labels=['Real', 'Synthetic'], loc='upper right')
    plt.savefig(save_path + "/isomap_component_matrix.png")
    plt.close()

    # l2 = euclidean(real_fpca_mean, synthetic_fpca_mean)
    # krzanowski = krzanowski_similarity(real_fpca_components, synthetic_fpca_components)
    # print("Real NORM vs Synthetic NORM")
    # print(f"L2 distance between real and synthetic mean: {l2}")
    # print(f"Krzanowski similarity between real and synthetic: {krzanowski}")

    # Get Synthetic Data of different diagnostic
    # synthetic_all = load_synthetic_dataset("MI", lead)
    # trimmed_synthetic_fd, synthetic_landmarks_all = extract_ecg_clinical_landmarks(synthetic_all, n_beats, sr)
    # synthetic_fd = trimmed_synthetic_fd[:n_data]
    # synthetic_landmarks = synthetic_landmarks_all[:n_data]
    
    # lambda_ = basis_smoothing_hyperparameter_tuning(synthetic_fd, n_basis, domain_range)
    # synthetic_smooth_fd, _, _, _ = basis_smoothing_with_lambda(synthetic_fd, lambda_, n_basis, domain_range)
    # synthetic_aligned_fd, _ = landmark_registration(synthetic_smooth_fd, synthetic_landmarks)
    # synthetic_fpca_mean, synthetic_fpca_components, synthetic_fpca_scores, synthetic_fpca_var_ratio, synthetic_fpca_ = fpca_with_param(synthetic_aligned_fd, n_components)
    # print(f"Variance ratio sum - Synthetic: {np.sum(synthetic_fpca_var_ratio)}")
    
    # synthetic_optimal_k = find_optimal_k(synthetic_fpca_scores)
    # synthetic_optimal_dim = find_optimal_manifold_dim(synthetic_fpca_scores, synthetic_optimal_k)
    # print(f"Optimal number of neighbors - Synthetic: {synthetic_optimal_k}")
    # print(f"Optimal number of components - Synthetic: {synthetic_optimal_dim}")

    # synthetic_iso = Isomap(n_neighbors=synthetic_optimal_k, n_components=synthetic_optimal_dim)
    # synthetic_embedding = synthetic_iso.fit_transform(synthetic_fpca_scores)

    # # Plot
    # shared_n = min(synthetic_optimal_dim, synthetic_optimal_dim)
    # synthetic_labels = np.ones(synthetic_embedding.shape[0]).reshape(-1, 1)
    # embeddings = np.concatenate((synthetic_embedding[:, :shared_n], synthetic_embedding[:, :shared_n]), axis=0)
    # labels = np.concatenate((synthetic_labels, synthetic_labels), axis=0)
    # df = pd.DataFrame(embeddings, columns=[f'Component {i+1}' for i in range(synthetic_optimal_dim)])
    # df['Source'] = labels
    # sns.pairplot(df, hue='Source', palette='viridis', diag_kind='kde', plot_kws={'alpha': 0.5})
    # plt.suptitle('Isomap Component Matrix: Real NORM vs Synthetic MI', y=1.02)
    # plt.savefig(save_path + "/isomap_component_matrix_mi.png")
    # plt.close()

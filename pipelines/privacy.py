import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from pathlib import Path
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks, load_synthetic_dataset
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.fpca import fpca_with_param
from methods.isomap import find_optimal_k, find_optimal_manifold_dim
from methods.privacy_domias import domias_vectorized_kde, domias_subspace_mahalanobis, full_knowledge_mia

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
    save_path = f"../images/privacy/"
    path=Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

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

    ### DOMIAS Density Ratio Privacy Evaluation###
    # Compute FPC Density Ratio #
    fpc_density_ratio = domias_subspace_mahalanobis(synthetic_scores, real_scores, holdout_fpca_scores)
    
    # Compute Isomap Embedding Density Ratio #
    isomap_density_ratio = domias_vectorized_kde(synthetic_embedding, real_embedding, holdout_embedding)

    # Proportion of Density Ratio > 1 #
    avg_fpc_privacy= np.mean(fpc_density_ratio > 1)
    avg_isomap_privacy = np.mean(isomap_density_ratio > 1)
    print(f"Proportion of FPC DOMIAS Density Ratio > 1: {avg_fpc_privacy}")
    print(f"Proportion of Isomap DOMIAS Density Ratio > 1: {avg_isomap_privacy}")

    # Density Ratio plot #
    log_fpc_density_ratio = np.log(fpc_density_ratio)
    log_isomap_density_ratio = np.log(isomap_density_ratio)

    plt.hist(log_fpc_density_ratio, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Log Density Ratio')
    plt.ylabel('Frequency (Count)')
    plt.title('Distribution of Log FPC Density Ratio')
    plt.savefig(save_path + 'fpc_density_ratio.png')

    plt.hist(log_isomap_density_ratio, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Log Density Ratio')
    plt.ylabel('Frequency (Count)')
    plt.title('Distribution of Log Isomap Density Ratio')
    plt.savefig(save_path + 'isomap_density_ratio.png')

    ### Full Knowledge MIA Privacy Evaluation ###
    real_knowledge = np.concatenate([real_scores, real_embedding], axis=1)
    synthetic_knowledge = np.concatenate([synthetic_scores, synthetic_embedding], axis=1)
    fpr, tpr, thresholds, mia_auc_roc = full_knowledge_mia(real_knowledge, synthetic_knowledge)
    print(f"The Full Knowledge MIA AUC-ROC is: {mia_auc_roc:.4f}")
    
    plt.figure(figsize=(7, 6))

    # Plot your custom model's curve
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'MIA (AUC = {mia_auc_roc:.3f})')

    # Plot the 0.50 Equilibrium baseline
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1.5, label='Perfect Equilibrium (AUC = 0.50)')

    # Visual Annotation Zones - Relocated to prevent high-AUC line overlap
    plt.text(0.60, 0.15, '⚠️ PRIVACY FAILURE\n(Real data memorized)', color='darkred', weight='bold', fontsize=9)
    plt.text(0.05, 0.85, '⚠️ FIDELITY FAILURE\n(Synthetic data looks fake)', color='darkorange', weight='bold', fontsize=9)
    plt.text(0.42, 0.48, '✨ SWEET SPOT', color='green', weight='bold', fontsize=9, rotation=37)

    # Formatting the axes with corrected inverted definitions
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (Real data classified as Synthetic)', fontsize=11)
    plt.ylabel('True Positive Rate (Synthetic classified correctly)', fontsize=11)
    plt.title('Tier 3 Privacy-Fidelity ROC Curve', fontsize=13, weight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path + 'full_knowledge_mia_roc_curve.png', dpi=300)
    plt.close()
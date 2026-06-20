import numpy as np
import tabulate as tb
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from pathlib import Path
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.transformation.fpca import fpca_with_param
from methods.transformation.kpca import tune_gamma, tuning_n_components, kpca_with_param
from methods.evaluation.privacy import domias, full_knowledge_mia
from methods.validation.data_creation import create_low_fidelity_dataset, create_mode_collapse_dataset, create_exact_memorization_dataset

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
    save_path = f"images/privacy_validation/"
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

    # Create Validation Datasets
    low_fidelity_dataset = create_low_fidelity_dataset(trimmed_real_fd)
    mode_collapse_dataset, mode_collapse_landmarks = create_mode_collapse_dataset(trimmed_real_fd, real_landmarks_all)
    exact_memorization_dataset, exact_memorization_landmarks = create_exact_memorization_dataset(trimmed_real_fd, real_landmarks_all)

    ### FPCA on Holdout Data ###
    # Apply FPCA on holdout dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(holdout_fd, n_basis, domain_range)
    holdout_fd_smooth, _, _, _ = basis_smoothing_with_lambda(holdout_fd, lambda_, n_basis, domain_range)
    holdout_aligned_fd, _ = landmark_registration(holdout_fd_smooth, holdout_landmarks, landmark_locations)
    holdout_fpca_mean, holdout_fpca_components, holdout_fpca_scores, holdout_fpca_var_ratio, holdout_fpca_ = fpca_with_param(holdout_aligned_fd, n_components)

    # Apply kPCA on holdout FPC scores
    optimal_gamma_holdout = tune_gamma(holdout_fpca_scores)
    optimal_dim_holdout = tuning_n_components(holdout_fpca_scores, optimal_gamma_holdout)
    holdout_embedding, holdout_kpca_ = kpca_with_param(holdout_fpca_scores, optimal_dim_holdout, optimal_gamma_holdout)

    # Apply Holdout FPCA and kPCA on Real dataset
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks, landmark_locations)
    real_scores = holdout_fpca_.transform(real_aligned_fd)
    real_embedding = holdout_kpca_.transform(real_scores)

    # Apply Holdout FPCA and kPCA on Validation Datasets
    low_fidelity_fd_smooth, _, _, _ = basis_smoothing_with_lambda(low_fidelity_dataset, lambda_, n_basis, domain_range)
    low_fidelity_aligned_fd, _ = landmark_registration(low_fidelity_fd_smooth, real_landmarks_all, landmark_locations)
    low_fidelity_scores = holdout_fpca_.transform(low_fidelity_aligned_fd)
    low_fidelity_embedding = holdout_kpca_.transform(low_fidelity_scores)
    
    mode_collapse_fd_smooth, _, _, _ = basis_smoothing_with_lambda(mode_collapse_dataset, lambda_, n_basis, domain_range)
    mode_collapse_aligned_fd, _ = landmark_registration(mode_collapse_fd_smooth, mode_collapse_landmarks, landmark_locations)
    mode_collapse_scores = holdout_fpca_.transform(mode_collapse_aligned_fd)
    mode_collapse_embedding = holdout_kpca_.transform(mode_collapse_scores)
    
    exact_memorization_fd_smooth, _, _, _ = basis_smoothing_with_lambda(exact_memorization_dataset, lambda_, n_basis, domain_range)
    exact_memorization_aligned_fd, _ = landmark_registration(exact_memorization_fd_smooth, exact_memorization_landmarks, landmark_locations)
    exact_memorization_scores = holdout_fpca_.transform(exact_memorization_aligned_fd)
    exact_memorization_embedding = holdout_kpca_.transform(exact_memorization_scores)

    #### DOMIAS Density Ratio Privacy Evaluation ####
    ## DOMIAS - Low Utility Dataset
    # Compute FPC Score and kPCA Embedding Density Ratio #
    low_fidelity_fpc_density_ratio = domias(holdout_fpca_scores, real_scores, low_fidelity_scores)
    low_fidelity_kPCA_density_ratio = domias(holdout_embedding, real_embedding, low_fidelity_embedding)

    bandwidth_grid = list(low_fidelity_fpc_density_ratio.keys())

    # FPC Score Results
    avg_fpc_privacy = []
    for bandwidth, score in low_fidelity_fpc_density_ratio.items():
        avg= np.mean(score > 0)
        avg_fpc_privacy.append(avg)

        plt.hist(score, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Log Density Ratio')
        plt.ylabel('Frequency (Count)')
        plt.title(f'Distribution of Log FPC Density Ratio (Bandwidth: {bandwidth:.3f})')
        plt.savefig(save_path + f'low_fidelity_fpc_density_ratio_{bandwidth:.3f}.png')
        plt.close()

    plt.plot(bandwidth_grid, avg_fpc_privacy)
    plt.xlabel('Bandwidth')
    plt.ylabel('Log Density Ratio')
    plt.title('Log Low Fidelity FPC Density Ratio vs. Kernel Bandwidth')
    plt.savefig(save_path + 'low_fidelity_fpc_density_ratio_vs_bandwidth.png')
    plt.close()
    
    # kPCA Embedding Density Ratio Results
    bandwidth_grid = list(low_fidelity_kPCA_density_ratio.keys())
    avg_kPCA_privacy = []
    for bandwidth, score in low_fidelity_kPCA_density_ratio.items():
        avg = np.mean(score > 0)
        avg_kPCA_privacy.append(avg)
    
        plt.hist(score, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Log Density Ratio')
        plt.ylabel('Frequency (Count)')
        plt.title(f'Distribution of Log kPCA Density Ratio (Bandwidth: {bandwidth:.3f})')
        plt.savefig(save_path + f'low_fidelity_kPCA_density_ratio_{bandwidth:.3f}.png')
        plt.close()
    
    plt.plot(bandwidth_grid, avg_kPCA_privacy)
    plt.xlabel('Bandwidth')
    plt.ylabel('Log Density Ratio')
    plt.title('Log Low Fidelity kPCA Density Ratio vs. Kernel Bandwidth')
    plt.savefig(save_path + 'low_fidelity_kPCA_density_ratio_vs_bandwidth.png')
    plt.close()

    ### Full Knowledge MIA Privacy Evaluation ###
    real_knowledge = np.concatenate([real_scores, real_embedding], axis=1)
    low_fidelity_knowledge = np.concatenate([low_fidelity_scores, low_fidelity_embedding], axis=1)
    fpr, tpr, thresholds, mia_auc_roc = full_knowledge_mia(real_knowledge, low_fidelity_knowledge)
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
    plt.title('Low Fidelity Privacy-Fidelity ROC Curve', fontsize=13, weight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path + 'low_fidelity_full_knowledge_mia_roc_curve.png', dpi=300)
    plt.close()

    ## DOMIAS - Mode Collapse Dataset
    # Compute FPC Score and kPCA Embedding Density Ratio #
    mode_collapse_fpc_density_ratio = domias(holdout_fpca_scores, real_scores, mode_collapse_scores)
    mode_collapse_kPCA_density_ratio = domias(holdout_embedding, real_embedding, mode_collapse_embedding)

    ### Result Display ###
    bandwidth_grid = list(mode_collapse_fpc_density_ratio.keys())

    # FPC Score Results
    avg_fpc_privacy = []
    for bandwidth, score in mode_collapse_fpc_density_ratio.items():
        avg= np.mean(score > 0)
        avg_fpc_privacy.append(avg)

        plt.hist(score, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Log Density Ratio')
        plt.ylabel('Frequency (Count)')
        plt.title(f'Distribution of Log FPC Density Ratio (Bandwidth: {bandwidth:.3f})')
        plt.savefig(save_path + f'mode_collapse_fpc_density_ratio_{bandwidth:.3f}.png')
        plt.close()

    plt.plot(bandwidth_grid, avg_fpc_privacy)
    plt.xlabel('Bandwidth')
    plt.ylabel('Log Density Ratio')
    plt.title('Log Mode Collapse FPC Density Ratio vs. Kernel Bandwidth')
    plt.savefig(save_path + 'mode_collapse_fpc_density_ratio_vs_bandwidth.png')
    plt.close()
    
    # kPCA Embedding Density Ratio Results
    bandwidth_grid = list(mode_collapse_kPCA_density_ratio.keys())
    avg_kPCA_privacy = []
    for bandwidth, score in mode_collapse_kPCA_density_ratio.items():
        avg = np.mean(score > 0)
        avg_kPCA_privacy.append(avg)
    
        plt.hist(score, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Log Density Ratio')
        plt.ylabel('Frequency (Count)')
        plt.title(f'Distribution of Log kPCA Density Ratio (Bandwidth: {bandwidth:.3f})')
        plt.savefig(save_path + f'mode_collapse_kPCA_density_ratio_{bandwidth:.3f}.png')
        plt.close()
    
    plt.plot(bandwidth_grid, avg_kPCA_privacy)
    plt.xlabel('Bandwidth')
    plt.ylabel('Log Density Ratio')
    plt.title('Log Mode Collapse kPCA Density Ratio vs. Kernel Bandwidth')
    plt.savefig(save_path + 'mode_collapse_kPCA_density_ratio_vs_bandwidth.png')
    plt.close()

    ### Full Knowledge MIA Privacy Evaluation ###
    real_knowledge = np.concatenate([real_scores, real_embedding], axis=1)
    mode_collapse_knowledge = np.concatenate([mode_collapse_scores, mode_collapse_embedding], axis=1)
    fpr, tpr, thresholds, mia_auc_roc = full_knowledge_mia(real_knowledge, mode_collapse_knowledge)
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
    plt.title('Mode Collapse Privacy-Fidelity ROC Curve', fontsize=13, weight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path + 'mode_collapse_full_knowledge_mia_roc_curve.png', dpi=300)
    plt.close()

    ## DOMIAS - Exact Memorization Dataset
    # Compute FPC Score and kPCA EmbeddingDensity Ratio #
    exact_memorization_fpc_density_ratio = domias(holdout_fpca_scores, real_scores, exact_memorization_scores)
    exact_memorization_kPCA_density_ratio = domias(holdout_embedding, real_embedding, exact_memorization_embedding)

    ### Result Display ###
    bandwidth_grid = list(exact_memorization_fpc_density_ratio.keys())

    # FPC Score Results
    avg_fpc_privacy = []
    for bandwidth, score in exact_memorization_fpc_density_ratio.items():
        avg= np.mean(score > 0)
        avg_fpc_privacy.append(avg)

        plt.hist(score, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Log Density Ratio')
        plt.ylabel('Frequency (Count)')
        plt.title(f'Distribution of Log FPC Density Ratio (Bandwidth: {bandwidth:.3f})')
        plt.savefig(save_path + f'exact_memorization_fpc_density_ratio_{bandwidth:.3f}.png')
        plt.close()

    plt.plot(bandwidth_grid, avg_fpc_privacy)
    plt.xlabel('Bandwidth')
    plt.ylabel('Log Density Ratio')
    plt.title('Log Exact Memorization FPC Density Ratio vs. Kernel Bandwidth')
    plt.savefig(save_path + 'exact_memorization_fpc_density_ratio_vs_bandwidth.png')
    plt.close()
    
    # kPCA Embedding Density Ratio Results
    bandwidth_grid = list(exact_memorization_kPCA_density_ratio.keys())
    avg_kPCA_privacy = []
    for bandwidth, score in exact_memorization_kPCA_density_ratio.items():
        avg = np.mean(score > 0)
        avg_kPCA_privacy.append(avg)
    
        plt.hist(score, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Log Density Ratio')
        plt.ylabel('Frequency (Count)')
        plt.title(f'Distribution of Log kPCA Density Ratio (Bandwidth: {bandwidth:.3f})')
        plt.savefig(save_path + f'exact_memorization_kPCA_density_ratio_{bandwidth:.3f}.png')
        plt.close()
    
    plt.plot(bandwidth_grid, avg_kPCA_privacy)
    plt.xlabel('Bandwidth')
    plt.ylabel('Log Density Ratio')
    plt.title('Log Exact Memorization kPCA Density Ratio vs. Kernel Bandwidth')
    plt.savefig(save_path + 'exact_memorization_kPCA_density_ratio_vs_bandwidth.png')
    plt.close()

    ### Full Knowledge MIA Privacy Evaluation ###
    real_knowledge = np.concatenate([real_scores, real_embedding], axis=1)
    exact_memorization_knowledge = np.concatenate([exact_memorization_scores, exact_memorization_embedding], axis=1)
    fpr, tpr, thresholds, mia_auc_roc = full_knowledge_mia(real_knowledge, exact_memorization_knowledge)
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
    plt.title('Exact Memorization Privacy-Fidelity ROC Curve', fontsize=13, weight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path + 'exact_memorization_full_knowledge_mia_roc_curve.png', dpi=300)
    plt.close()
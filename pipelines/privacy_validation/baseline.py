import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.evaluation.privacy import full_knowledge_mia
from methods.validation.dataset_creation import *

def dsintace_to_closest_record(real_aligned_fd, flaw_aligned_fd):
    # For each flaw record, find its distance to the closest real record
    distances = []
    for flaw_record in flaw_aligned_fd:
        closest_real_record = min(real_aligned_fd, key=lambda x: np.linalg.norm(x - flaw_record))
        distances.append(np.linalg.norm(closest_real_record - flaw_record))
    return distances

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
    save_path = f"images/privacy_val/baseline/"
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
    holdout_idx = sampled_idx[n_data:2*n_data]

    real_fd = trimmed_real_fd[real_idx]
    real_landmarks = real_landmarks_all[real_idx]
    holdout_fd = trimmed_real_fd[holdout_idx]
    holdout_landmarks = real_landmarks_all[holdout_idx]

    # Registration of Holdout dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(holdout_fd, n_basis, domain_range)
    holdout_fd_smooth, _, _, _ = basis_smoothing_with_lambda(holdout_fd, lambda_, n_basis, domain_range)
    holdout_aligned_fd, _ = landmark_registration(holdout_fd_smooth, holdout_landmarks, landmark_locations)
    
    # Registration of Real dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(trimmed_real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(trimmed_real_fd, lambda_, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks_all, landmark_locations)

    scenarios = ["oversmoothing", "memorization", "gaussian_noise", "mode_collapse_vary_modes", "mode_collapse_vary_spike_ratio", "segment_leaking"]
    datasets = {}
    for scenario in scenarios:
        if scenario == "oversmoothing":
            datasets = oversmoothing_creation(real_fd, real_landmarks)
        elif scenario == "memorization":
            datasets = memorization_creation(real_fd, holdout_fd, real_landmarks, holdout_landmarks)
        elif scenario == "gaussian_noise":
            datasets = gaussian_noise_creation(real_fd, real_landmarks)
        elif scenario == "mode_collapse_vary_modes":
            datasets = mode_collapse_vary_modes_creation(real_fd, real_landmarks)
        elif scenario == "mode_collapse_vary_spike_ratio":
            datasets = mode_collapse_vary_spike_ratio_creation(real_fd, real_landmarks)
        elif scenario == "segment_leaking":
            datasets = segment_leaking_creation(real_fd, holdout_fd, real_landmarks, holdout_landmarks)

        fprs, tprs, roc_aucs, scales = [], [], [], []
        for key, value in datasets.items():
            flaw_fd, flaw_landmarks = value

            # Registration of Flaw dataset
            lambda_ = basis_smoothing_hyperparameter_tuning(flaw_fd, n_basis, domain_range)
            flaw_fd_smooth, _, _, _ = basis_smoothing_with_lambda(flaw_fd, lambda_, n_basis, domain_range)
            flaw_aligned_fd, _ = landmark_registration(flaw_fd_smooth, flaw_landmarks, landmark_locations)

            fpr, tpr, thresholds, mia_auc_roc = full_knowledge_mia(real_aligned_fd.data_matrix.squeeze(), flaw_aligned_fd.data_matrix.squeeze())
            fprs.append(fpr)
            tprs.append(tpr)
            roc_aucs.append(mia_auc_roc)
            scales.append(key)
            print(f"The Full-Knowledge MIA AUC-ROC is: {mia_auc_roc:.4f}")

        plt.figure(figsize=(7, 6))
        for fpr, tpr, roc_auc, scale in zip(fprs, tprs, roc_aucs, scales):
            plt.plot(fpr, tpr, lw=2.5, label=f'MIA (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1.5, label='Perfect Equilibrium (AUC = 0.50)')
        plt.text(0.60, 0.15, '⚠️ PRIVACY FAILURE\n(Real data memorized)', color='darkred', weight='bold', fontsize=9)
        plt.text(0.05, 0.85, '⚠️ FIDELITY FAILURE\n(Synthetic data looks fake)', color='darkorange', weight='bold', fontsize=9)
        plt.text(0.42, 0.48, '✨ SWEET SPOT', color='green', weight='bold', fontsize=9, rotation=37)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate (Real data classified as Synthetic)', fontsize=11)
        plt.ylabel('True Positive Rate (Synthetic classified correctly)', fontsize=11)
        plt.title(f'Baseline Privacy-Fidelity ROC Curve ({scenario})', fontsize=13, weight='bold', pad=15)
        plt.legend(loc="lower right", frameon=True, shadow=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path + f'Baseline_roc_curve_{scenario}.png', dpi=300)
        plt.close()

            # member_distances = dsintace_to_closest_record(real_aligned_fd.data_matrix.squeeze(), flaw_aligned_fd.data_matrix.squeeze())
            # non_member_distances = dsintace_to_closest_record(holdout_aligned_fd.data_matrix.squeeze(), flaw_aligned_fd.data_matrix.squeeze())

            # # ==========================================
            # # 4. Evaluate Attack Success (ROC-AUC)
            # # ==========================================
            # # Ground truth labels: 1 for Member, 0 for Non-Member
            # y_true = np.concatenate([np.ones_like(member_distances), np.zeros_like(non_member_distances)])

            # # By definition, smaller distance = higher chance of being a member.
            # # We negate the distances so that higher scores correlate with membership label 1.
            # y_scores = -np.concatenate([member_distances, non_member_distances])

            # fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            # auc_score = roc_auc_score(y_true, y_scores)

            # plt.figure(figsize=(7, 6))
            # plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'DTW-MIA (AUC = {auc_score:.3f})')
            # plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1.5, label='Chance (AUC = 0.50)')
            # plt.xlim([-0.02, 1.02])
            # plt.ylim([-0.02, 1.02])
            # plt.xlabel('False Positive Rate', fontsize=11)
            # plt.ylabel('True Positive Rate', fontsize=11)
            # plt.title(f'DTW-MIA ROC Curve ({scenario}, {key})', fontsize=13, weight='bold', pad=15)
            # plt.legend(loc="lower right", frameon=True, shadow=True)
            # plt.grid(True, linestyle=':', alpha=0.6)
            # plt.tight_layout()
            # plt.savefig(save_path + f'baseline_roc_curve_{scenario}_{key}.png', dpi=300)
            # plt.close()
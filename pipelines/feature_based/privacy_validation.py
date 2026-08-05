import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks
from preprocess.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from transformation.fda.fpca import fpca_with_param
from transformation.nonlinear.diffusion_map import DenseDiffusionMap
from metrics.privacy import *
from validation.dataset_creation import *

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
    save_path = f"images/privacy_val/fpca/"
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

    # Create Controlled Flaw Dataset
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

        # FULL-knowledge MIA tracks
        fprs, tprs, roc_aucs, scales = [], [], [], []
        for key, value in datasets.items():
            flaw_fd, flaw_landmarks = value
            #### ------------ Shared FPCA ------------ ####
            # Apply FPCA on Holdout dataset
            lambda_ = basis_smoothing_hyperparameter_tuning(holdout_fd, n_basis, domain_range)
            holdout_fd_smooth, _, _, _ = basis_smoothing_with_lambda(holdout_fd, lambda_, n_basis, domain_range)
            holdout_aligned_fd, _ = landmark_registration(holdout_fd_smooth, holdout_landmarks, landmark_locations)
            holdout_mean, holdout_components, holdout_scores, holdout_var_ratio, holdout_fpca_ = fpca_with_param(holdout_aligned_fd, n_components)

            # Apply Holdout FPCA on Real dataset
            lambda_ = basis_smoothing_hyperparameter_tuning(real_fd, n_basis, domain_range)
            real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
            real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks, landmark_locations)
            real_scores = holdout_fpca_.transform(real_aligned_fd)
            
            # Apply Holdout FPCA on flaw dataset
            lambda_ = basis_smoothing_hyperparameter_tuning(flaw_fd, n_basis, domain_range)
            flaw_fd_smooth, _, _, _ = basis_smoothing_with_lambda(flaw_fd, lambda_, n_basis, domain_range)
            flaw_aligned_fd, _ = landmark_registration(flaw_fd_smooth, flaw_landmarks, landmark_locations)
            flaw_scores = holdout_fpca_.transform(flaw_aligned_fd)

            # Evaluation: DOMIAS on FPCA scores
            # fpc_density_ratio = domias(holdout_scores, real_scores, flaw_scores)

            # Apply Diffusion Map on holdout FPC scores
            holdout_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(holdout_scores)
            holdout_dmap_evals = holdout_dmap.evals_
            holdout_dmap_embedding = holdout_dmap.transform(holdout_scores)

            # Apply holdout diffusion map on real FPC scores
            real_dmap_embedding = holdout_dmap.transform(real_scores)

            # Apply holdout diffusion map on flaw FPC scores
            flaw_dmap_embedding = holdout_dmap.transform(flaw_scores)

            ## Evaluation: DOMIAS on Diffusion map embeddings
            dmap_density_ratio = domias(holdout_dmap_embedding, real_dmap_embedding, flaw_dmap_embedding)

            # Apply UMAP on holdout FPC scores
            # holdout_umap = tune_umap(holdout_scores)
            # holdout_umap_embedding = holdout_umap.transform(holdout_scores)

            # # Apply holdout UMAP on real FPC scores
            # real_umap_embedding = holdout_umap.transform(real_scores)

            # # Apply holdout UMAP on flaw FPC scores
            # flaw_umap_embedding = holdout_umap.transform(flaw_scores)

            # ## Evaluation: DOMIAS on UMAP embeddings
            # umap_density_ratio = domias(holdout_umap_embedding, real_umap_embedding, flaw_umap_embedding)

            # # Apply kPCA on holdout FPC scores
            # holdout_kpca_n_components = kpca_tune_n_components(holdout_scores)
            # holdout_kpca_gamma = tune_gamma(holdout_scores)
            # holdout_kpca_embedding, holdout_kpca = kpca_with_param(holdout_scores, holdout_kpca_n_components, holdout_kpca_gamma)

            # # Apply holdout kPCA on real FPC scores
            # real_kpca_embedding = holdout_kpca.transform(real_scores)

            # # Apply holdout kPCA on flaw FPC scores
            # flaw_kpca_embedding = holdout_kpca.transform(flaw_scores)

            # ## Evaluation: DOMIAS on kPCA embeddings
            # kpca_density_ratio = domias(holdout_kpca_embedding, real_kpca_embedding, flaw_kpca_embedding)

            #### ------------ Result Display ------------ ####
            # bandwidth_grid = list(fpc_density_ratio.keys())
            # # FPCA DOMIAS
            # avg_fpc_privacy = []
            # for bandwidth, score in fpc_density_ratio.items():
            #     avg= np.mean(score > 0)
            #     avg_fpc_privacy.append(avg)

            #     plt.hist(score, bins=50, color='skyblue', edgecolor='black')
            #     plt.xlabel('Log Density Ratio')
            #     plt.ylabel('Frequency (Count)')
            #     plt.title(f'Distribution of Log FPC Density Ratio (Bandwidth: {bandwidth:.3f})')
            #     plt.savefig(save_path + f'FPCA_density_ratio_{scenario}_{key}_{bandwidth:.3f}.png')
            #     plt.close()

            # plt.plot(bandwidth_grid, avg_fpc_privacy)
            # plt.xlabel('Bandwidth')
            # plt.ylabel('Log Density Ratio')
            # plt.title('Log FPC Density Ratio vs. Kernel Bandwidth')
            # plt.savefig(save_path + f'FPCA_density_ratio_vs_bandwidth_{scenario}_{key}.png')
            # plt.close()

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
                plt.savefig(save_path + f'DMap_density_ratio_{scenario}_{key}_{bandwidth:.3f}.png')
                plt.close()
            
            plt.plot(bandwidth_grid, avg_dmap_privacy)
            plt.xlabel('Bandwidth')
            plt.ylabel('Log Density Ratio')
            plt.title('Log Diffusion Map Density Ratio vs. Kernel Bandwidth')
            plt.savefig(save_path + f'DMap_density_ratio_vs_bandwidth_{scenario}_{key}.png')
            plt.close()

            # UMAP DOMIAS
            # bandwidth_grid = list(umap_density_ratio.keys())
            # avg_umap_privacy = []
            # for bandwidth, score in umap_density_ratio.items():
            #     avg = np.mean(score > 0)
            #     avg_umap_privacy.append(avg)

            #     plt.hist(score, bins=50, color='skyblue', edgecolor='black')
            #     plt.xlabel('Log Density Ratio')
            #     plt.ylabel('Frequency (Count)')
            #     plt.title(f'Distribution of Log UMAP Density Ratio (Bandwidth: {bandwidth:.3f})')
            #     plt.savefig(save_path + f'UMAP_density_ratio_{scenario}_{key}_{bandwidth:.3f}.png')
            #     plt.close()
            
            # plt.plot(bandwidth_grid, avg_umap_privacy)
            # plt.xlabel('Bandwidth')
            # plt.ylabel('Log Density Ratio')
            # plt.title('Log UMAP Density Ratio vs. Kernel Bandwidth')
            # plt.savefig(save_path + f'UMAP_density_ratio_vs_bandwidth_{scenario}_{key}.png')
            # plt.close()

            # # kPCA DOMIAS
            # bandwidth_grid = list(kpca_density_ratio.keys())
            # avg_kpca_privacy = []
            # for bandwidth, score in kpca_density_ratio.items():
            #     avg = np.mean(score > 0)
            #     avg_kpca_privacy.append(avg)

            #     plt.hist(score, bins=50, color='skyblue', edgecolor='black')
            #     plt.xlabel('Log Density Ratio')
            #     plt.ylabel('Frequency (Count)')
            #     plt.title(f'Distribution of Log kPCA Density Ratio (Bandwidth: {bandwidth:.3f})')
            #     plt.savefig(save_path + f'kPCA_density_ratio_{scenario}_{key}_{bandwidth:.3f}.png')
            #     plt.close()
            
            # plt.plot(bandwidth_grid, avg_kpca_privacy)
            # plt.xlabel('Bandwidth')
            # plt.ylabel('Log Density Ratio')
            # plt.title('Log kPCA Density Ratio vs. Kernel Bandwidth')
            # plt.savefig(save_path + f'kPCA_density_ratio_vs_bandwidth_{scenario}_{key}.png')
            # plt.close()

            # Full-Knowledge MIA: FPCA + Diffusion Map
            real_fpca_dmap = np.concatenate([real_scores, real_dmap_embedding], axis=1)
            flaw_fpca_dmap = np.concatenate([flaw_scores, flaw_dmap_embedding], axis=1)
            fpr, tpr, thresholds, mia_auc_roc = full_knowledge_mia(real_fpca_dmap, flaw_fpca_dmap)
            print(f"The Full-Knowledge MIA AUC-ROC is: {mia_auc_roc:.4f}")
            fprs.append(fpr)
            tprs.append(tpr)
            roc_aucs.append(mia_auc_roc)
            scales.append(key)

        plt.figure(figsize=(7, 6))
        for fpr, tpr, roc_auc, scale in zip(fprs, tprs, roc_aucs, scales):
            plt.plot(fpr, tpr, lw=2.5, label=f'Scale: {scale}, MIA (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1.5, label='Perfect Equilibrium (AUC = 0.50)')
        plt.text(0.60, 0.15, '⚠️ PRIVACY FAILURE\n(Real data memorized)', color='darkred', weight='bold', fontsize=9)
        plt.text(0.05, 0.85, '⚠️ FIDELITY FAILURE\n(Synthetic data looks fake)', color='darkorange', weight='bold', fontsize=9)
        plt.text(0.42, 0.48, '✨ SWEET SPOT', color='green', weight='bold', fontsize=9, rotation=37)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate (Real data classified as Synthetic)', fontsize=11)
        plt.ylabel('True Positive Rate (Synthetic classified correctly)', fontsize=11)
        plt.title(f'FPCA + Diffusion Map Privacy-Fidelity ROC Curve ({scenario})', fontsize=13, weight='bold', pad=15)
        plt.legend(loc="lower right", frameon=True, shadow=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path + f'FPCA_DMap_full_knowledge_mia_roc_curve_{scenario}.png', dpi=300)
        plt.close()

            # # Full-Knowledge MIA: FPCA + UMAP
            # real_fpca_umap = np.concatenate([real_scores, real_umap_embedding], axis=1)
            # flaw_fpca_umap = np.concatenate([flaw_scores, flaw_umap_embedding], axis=1)
            # fpr, tpr, thresholds, mia_auc_roc = full_knowledge_mia(real_fpca_umap, flaw_fpca_umap)
            # print(f"The Full-Knowledge MIA AUC-ROC is: {mia_auc_roc:.4f}")

            # plt.figure(figsize=(7, 6))
            # plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'MIA (AUC = {mia_auc_roc:.3f})')
            # plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1.5, label='Perfect Equilibrium (AUC = 0.50)')
            # plt.text(0.60, 0.15, '⚠️ PRIVACY FAILURE\n(Real data memorized)', color='darkred', weight='bold', fontsize=9)
            # plt.text(0.05, 0.85, '⚠️ FIDELITY FAILURE\n(Synthetic data looks fake)', color='darkorange', weight='bold', fontsize=9)
            # plt.text(0.42, 0.48, '✨ SWEET SPOT', color='green', weight='bold', fontsize=9, rotation=37)
            # plt.xlim([-0.02, 1.02])
            # plt.ylim([-0.02, 1.02])
            # plt.xlabel('False Positive Rate (Real data classified as Synthetic)', fontsize=11)
            # plt.ylabel('True Positive Rate (Synthetic classified correctly)', fontsize=11)
            # plt.title('FPCA + UMAP Privacy-Fidelity ROC Curve', fontsize=13, weight='bold', pad=15)
            # plt.legend(loc="lower right", frameon=True, shadow=True)
            # plt.grid(True, linestyle=':', alpha=0.6)
            # plt.tight_layout()
            # plt.savefig(save_path + f'fpca_umap_full_knowledge_mia_roc_curve_{scenario}_{key}.png', dpi=300)
            # plt.close()

            # # Full-Knowledge MIA: FPCA + kPCA
            # real_fpca_kpca = np.concatenate([real_scores, real_kpca_embedding], axis=1)
            # flaw_fpca_kpca = np.concatenate([flaw_scores, flaw_kpca_embedding], axis=1)
            # fpr, tpr, thresholds, mia_auc_roc = full_knowledge_mia(real_fpca_kpca, flaw_fpca_kpca)
            # print(f"The Full-Knowledge MIA AUC-ROC is: {mia_auc_roc:.4f}")

            # plt.figure(figsize=(7, 6))
            # plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'MIA (AUC = {mia_auc_roc:.3f})')
            # plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1.5, label='Perfect Equilibrium (AUC = 0.50)')
            # plt.text(0.60, 0.15, '⚠️ PRIVACY FAILURE\n(Real data memorized)', color='darkred', weight='bold', fontsize=9)
            # plt.text(0.05, 0.85, '⚠️ FIDELITY FAILURE\n(Synthetic data looks fake)', color='darkorange', weight='bold', fontsize=9)
            # plt.text(0.42, 0.48, '✨ SWEET SPOT', color='green', weight='bold', fontsize=9, rotation=37)
            # plt.xlim([-0.02, 1.02])
            # plt.ylim([-0.02, 1.02])
            # plt.xlabel('False Positive Rate (Real data classified as Synthetic)', fontsize=11)
            # plt.ylabel('True Positive Rate (Synthetic classified correctly)', fontsize=11)
            # plt.title('FPCA + kPCA Privacy-Fidelity ROC Curve', fontsize=13, weight='bold', pad=15)
            # plt.legend(loc="lower right", frameon=True, shadow=True)
            # plt.grid(True, linestyle=':', alpha=0.6)
            # plt.tight_layout()
            # plt.savefig(save_path + f'fpca_kpca_full_knowledge_mia_roc_curve_{scenario}_{key}.png', dpi=300)
            # plt.close()
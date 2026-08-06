import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess.ptbxl_preprocess import load_dataset, get_sr, extract_ecg_phase_aligned
from preprocess.fpca_preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda
from transformation.fda.fpca import fpca_with_param
from transformation.nonlinear.diffusion_map import DenseDiffusionMap
from metrics.privacy import *
from scenario_engineering.dataset_creation import *

if __name__ == "__main__":
    ## ------------ Data Preparation ------------ ##
    diagnostic = "NORM"
    lead = 1
    sr = get_sr()
    domain_range = (0, 1)
    n_components = 10

    # Result save path
    save_path = f"images/privacy_val/fpca/"
    path=Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Get Real Data
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    aligned_real_fd = extract_ecg_phase_aligned(real_all, sr)
    n_sample, n_timepoints, n_channel = aligned_real_fd.data_matrix.shape
    n_basis = int(n_timepoints / 2)

    real_fd = aligned_real_fd[:n_sample//2]
    holdout_fd = aligned_real_fd[n_sample//2:]

    # Create Controlled Flaw Dataset
    scenarios = ["oversmoothing", "memorization", "gaussian_noise", "mode_collapse_vary_modes", "mode_collapse_vary_spike_ratio", "segment_leaking"]
    datasets = {}
    for scenario in scenarios:
        with open(save_path + f"{scenario}_dataset.pkl", "rb") as f:
            datasets = pickle.load(f)

        # FULL-knowledge MIA tracks
        fprs, tprs, roc_aucs, scales = [], [], [], []
        for key, value in datasets.items():
            flaw_fd = value
            #### ------------ Shared FPCA ------------ ####
            # Apply FPCA on Holdout dataset
            lambda_ = basis_smoothing_hyperparameter_tuning(holdout_fd, n_basis, domain_range)
            holdout_fd_smooth, _, _, _ = basis_smoothing_with_lambda(holdout_fd, lambda_, n_basis, domain_range)
            holdout_mean, holdout_components, holdout_scores, holdout_var_ratio, holdout_fpca_ = fpca_with_param(holdout_fd_smooth, n_components)

            # Apply Holdout FPCA on Real dataset
            lambda_ = basis_smoothing_hyperparameter_tuning(real_fd, n_basis, domain_range)
            real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
            real_scores = holdout_fpca_.transform(real_fd_smooth)
            
            # Apply Holdout FPCA on flaw dataset
            lambda_ = basis_smoothing_hyperparameter_tuning(flaw_fd, n_basis, domain_range)
            flaw_fd_smooth, _, _, _ = basis_smoothing_with_lambda(flaw_fd, lambda_, n_basis, domain_range)
            flaw_scores = holdout_fpca_.transform(flaw_fd_smooth)

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

            # Full-Knowledge MIA: FPCA + Diffusion Map
            real_fpca_dmap = np.concatenate([real_scores, real_dmap_embedding], axis=1)
            flaw_fpca_dmap = np.concatenate([flaw_scores, flaw_dmap_embedding], axis=1)
            fpr, tpr, thresholds, mia_auc_roc = classifier_mia(real_fpca_dmap, flaw_fpca_dmap)
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
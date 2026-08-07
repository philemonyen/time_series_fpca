import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from preprocess.ptbxl_preprocess import load_dataset, get_sr, extract_ecg_phase_aligned
from preprocess.fpca_preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda
from transformation.fda.fpca import fpca_with_param
from transformation.nonlinear.diffusion_map import DenseDiffusionMap
from metrics.privacy import *
from scenario_engineering.dataset_creation import *

def plot_roc_curve(fprs, tprs, roc_aucs, scales, name, save_path):
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
        plt.title(f'{name} ROC Curve', fontsize=13, weight='bold', pad=15)
        plt.legend(loc="lower right", frameon=True, shadow=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path + f'{name}_ROC_Curve.png', dpi=300)
        plt.close()


if __name__ == "__main__":
    ## ------------ Data Preparation ------------ ##
    diagnostic = "NORM"
    lead = 1
    sr = get_sr()
    domain_range = (0, 1)
    n_components = 10

    np.random.seed(42)

    # Get Real Data
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    aligned_real_fd = extract_ecg_phase_aligned(real_all, sr)
    n_sample, n_timepoints, n_channel = aligned_real_fd.data_matrix.shape
    n_basis = int(n_timepoints / 2)

    real_fd = aligned_real_fd[:n_sample//2]
    holdout_fd = aligned_real_fd[n_sample//2:]

    lambda_ = basis_smoothing_hyperparameter_tuning(holdout_fd, n_basis, domain_range)
    holdout_fd_smooth, _, _, _ = basis_smoothing_with_lambda(holdout_fd, lambda_, n_basis, domain_range)
    holdout_mean, holdout_components, holdout_scores, holdout_var_ratio, holdout_fpca_ = fpca_with_param(holdout_fd_smooth, n_components)

    lambda_ = basis_smoothing_hyperparameter_tuning(real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
    real_scores = holdout_fpca_.transform(real_fd_smooth)

    holdout_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(holdout_scores)
    holdout_dmap_evals = holdout_dmap.evals_
    holdout_dmap_embedding = holdout_dmap.transform(holdout_scores)
    real_dmap_embedding = holdout_dmap.transform(real_scores)

    # Create Controlled Flaw Dataset
    scenarios = ["oversmoothing", "memorization", "gaussian_noise", "mode_collapse_vary_modes", "mode_collapse_vary_spike_ratio", "segment_leaking"]
    datasets = {}
    for scenario in scenarios:
        # Result save path
        save_path = f"images/privacy_val/{scenario}/"
        path=Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        with open(f"data/validation/{scenario}_dataset.pkl", "rb") as f:
            datasets = pickle.load(f)

        # Result Tracking
        scales = []
        baseline_fprs, baseline_tprs, baseline_roc_aucs = [], [], []
        fpc_domias_fprs, fpc_domias_tprs, fpc_domias_roc_aucs = [], [], []
        dmap_domias_fprs, dmap_domias_tprs, dmap_domias_roc_aucs = [], [], []
        fpc_classifier_fprs, fpc_classifier_tprs, fpc_classifier_roc_aucs = [], [], []
        dmap_classifier_fprs, dmap_classifier_tprs, dmap_classifier_roc_aucs = [], [], []
        fpc_dmap_classifier_fprs, fpc_dmap_classifier_tprs, fpc_dmap_classifier_roc_aucs = [], [], []


        for key, flaw_fd in datasets.items():
            scales.append(key)
            #### ------------ Transformations ------------ ####
            # FPCA
            lambda_ = basis_smoothing_hyperparameter_tuning(flaw_fd, n_basis, domain_range)
            flaw_fd_smooth, _, _, _ = basis_smoothing_with_lambda(flaw_fd, lambda_, n_basis, domain_range)
            flaw_scores = holdout_fpca_.transform(flaw_fd_smooth)

            # Diffusion Map
            flaw_dmap_embedding = holdout_dmap.transform(flaw_scores)

            #### ------------ Evaluation ------------ ####
            # Raw Time-seriesDCR baseline
            dcr_baseline = dcr(holdout_fd.data_matrix.squeeze(), real_fd.data_matrix.squeeze(), flaw_fd.data_matrix.squeeze())
            fpr, tpr, thresholds, mia_auc_roc = dcr_mia(holdout_fd.data_matrix.squeeze(), real_fd.data_matrix.squeeze(), flaw_fd.data_matrix.squeeze())
            baseline_fprs.append(fpr)
            baseline_tprs.append(tpr)
            baseline_roc_aucs.append(mia_auc_roc)

            # DOMIAS MIA on FPCA scores
            fpc_density_ratio = domias(holdout_scores, real_scores, flaw_scores)
            fpr, tpr, thresholds, mia_auc_roc = domias_mia(holdout_scores, real_scores, flaw_scores)
            fpc_domias_fprs.append(fpr)
            fpc_domias_tprs.append(tpr)
            fpc_domias_roc_aucs.append(mia_auc_roc)

            # Classifier MIA on FPCA scores
            fpr, tpr, thresholds, mia_auc_roc = classifier_mia(real_scores, flaw_scores)
            fpc_classifier_fprs.append(fpr)
            fpc_classifier_tprs.append(tpr)
            fpc_classifier_roc_aucs.append(mia_auc_roc)

            # DOMIAS MIA on Diffusion map embeddings
            dmap_density_ratio = domias(holdout_dmap_embedding, real_dmap_embedding, flaw_dmap_embedding)
            fpr, tpr, thresholds, mia_auc_roc = domias_mia(holdout_dmap_embedding, real_dmap_embedding, flaw_dmap_embedding)
            dmap_domias_fprs.append(fpr)
            dmap_domias_tprs.append(tpr)
            dmap_domias_roc_aucs.append(mia_auc_roc)
            
            # Classifier MIA on Diffusion map embeddings
            fpr, tpr, thresholds, mia_auc_roc = classifier_mia(real_dmap_embedding, flaw_dmap_embedding)
            dmap_classifier_fprs.append(fpr)
            dmap_classifier_tprs.append(tpr)
            dmap_classifier_roc_aucs.append(mia_auc_roc)

            # Classifier MIA on FPCA + Diffusion map embeddings
            real_fpca_dmap = np.concatenate([real_scores, real_dmap_embedding], axis=1)
            flaw_fpca_dmap = np.concatenate([flaw_scores, flaw_dmap_embedding], axis=1)
            scaler = StandardScaler()
            scaled_real = scaler.fit_transform(real_fpca_dmap)
            scaled_flaw = scaler.transform(flaw_fpca_dmap)
            fpr, tpr, thresholds, mia_auc_roc = classifier_mia(scaled_real, scaled_flaw)
            fpc_dmap_classifier_fprs.append(fpr)
            fpc_dmap_classifier_tprs.append(tpr)
            fpc_dmap_classifier_roc_aucs.append(mia_auc_roc)

        plot_roc_curve(baseline_fprs, baseline_tprs, baseline_roc_aucs, scales, f'DCR Baseline ({scenario})', save_path)
        plot_roc_curve(fpc_domias_fprs, fpc_domias_tprs, fpc_domias_roc_aucs, scales, f'FPC DOMIAS MIA ({scenario})', save_path)
        plot_roc_curve(fpc_classifier_fprs, fpc_classifier_tprs, fpc_classifier_roc_aucs, scales, f'FPC Classifier MIA ({scenario})', save_path)
        plot_roc_curve(dmap_domias_fprs, dmap_domias_tprs, dmap_domias_roc_aucs, scales, f'DMap DOMIAS MIA ({scenario})', save_path)
        plot_roc_curve(dmap_classifier_fprs, dmap_classifier_tprs, dmap_classifier_roc_aucs, scales, f'DMap Classifier MIA ({scenario})', save_path)
        plot_roc_curve(fpc_dmap_classifier_fprs, fpc_dmap_classifier_tprs, fpc_dmap_classifier_roc_aucs, scales, f'FPC + DMap Classifier MIA ({scenario})', save_path)
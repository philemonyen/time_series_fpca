import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from preprocess.ptbxl_preprocess import load_dataset, get_sr, extract_ecg_sliding_windows
from preprocess.fpca_preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from transformation.fda.fpca import fpca_hyperparameter_tuning, fpca_with_param
from transformation.nonlinear.diffusion_map import DenseDiffusionMap
from metrics.privacy import *

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
    n_components = 10
    domain_range = (0, 1)
    np.random.seed(42)

    # Get real warping functions and apply FPCA
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    segments, landmarks = extract_ecg_sliding_windows(real_all, sr)
    n_data = segments.data_matrix.shape[0]
    real_segments = segments[:n_data//2]
    real_landmarks = landmarks[:n_data//2]
    holdout_segments = segments[n_data//2:]
    holdout_landmarks = landmarks[n_data//2:]

    aligned_holdout, holdout_warping_ = landmark_registration(holdout_segments, holdout_landmarks)
    n_basis = int(holdout_warping_.data_matrix.shape[1] / 2)
    lambda_ = basis_smoothing_hyperparameter_tuning(holdout_warping_, n_basis, domain_range)
    smoothed_holdout, _, _, _ = basis_smoothing_with_lambda(holdout_warping_, lambda_, n_basis, domain_range)
    n_components = fpca_hyperparameter_tuning(smoothed_holdout)
    holdout_mean, holdout_components, holdout_scores, holdout_var_ratio, holdout_fpca_ = fpca_with_param(smoothed_holdout, n_components)

    aligned_real, real_warping_ = landmark_registration(real_segments, real_landmarks)
    n_basis = int(real_warping_.data_matrix.shape[1] / 2)
    lambda_ = basis_smoothing_hyperparameter_tuning(real_warping_, n_basis, domain_range)
    smoothed_real, _, _, _ = basis_smoothing_with_lambda(real_warping_, lambda_, n_basis, domain_range)
    real_scores = holdout_fpca_.transform(smoothed_real)

    holdout_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(holdout_scores)
    holdout_dmap_evals = holdout_dmap.evals_
    holdout_dmap_embedding = holdout_dmap.transform(holdout_scores)
    real_dmap_embedding = holdout_dmap.transform(real_scores)

    # Load Controlled Flaw Dataset
    scenarios = ["phase_shift", "time_distortion"]
    datasets = {}
    result_tracking = {}
    for scenario in scenarios:
        save_path = f"images/fidelity_val/temporal/{scenario}/"
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


        for key, value in datasets.items():
            flaw_fd, landmarks = value

            #### ------------ Transformations ------------ ####
            # FPCA
            aligned_flaw, flaw_warping_ = landmark_registration(flaw_fd, landmarks)
            n_basis = int(flaw_warping_.data_matrix.shape[1] / 2)
            lambda_ = basis_smoothing_hyperparameter_tuning(flaw_warping_, n_basis, domain_range)
            smoothed_flaw, _, _, _ = basis_smoothing_with_lambda(flaw_warping_, lambda_, n_basis, domain_range)
            flaw_scores = holdout_fpca_.transform(smoothed_flaw)

            # Diffusion Map
            flaw_dmap_embedding = holdout_dmap.transform(flaw_scores)

            #### ------------ Evaluation ------------ ####
            # Raw Time-seriesDCR baseline
            dcr_baseline = dcr(holdout_warping_.data_matrix.squeeze(), real_warping_.data_matrix.squeeze(), flaw_warping_.data_matrix.squeeze())
            fpr, tpr, thresholds, mia_auc_roc = dcr_mia(holdout_warping_.data_matrix.squeeze(), real_warping_.data_matrix.squeeze(), flaw_warping_.data_matrix.squeeze())
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
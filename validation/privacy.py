import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from preprocess.ptbxl_preprocess import align_ecg
from preprocess.fpca_preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda
from transformation.fda.fpca import fpca_with_param
from transformation.nonlinear.diffusion_map import DenseDiffusionMap
from transformation.baseline.pca import *
from transformation.baseline.fft import *
from transformation.baseline.wavelet import *
from scenario_engineering.controlled_flaw_modelling import inject_segment_leak
from metrics.privacy import *


def find_highly_unique_target_records(real_data, n=5):
    """
    Extract the top n most unique target records from the real data according to Mahalanobis distance.
    Args:
        real_data (numpy.ndarray): The real data.
        n (int): The number of most unique target records to extract.
    Returns:
        numpy.ndarray: The top n most unique target records.
    """
    X = np.asarray(real_data, dtype=float)
    n_samples = X.shape[0]
    n = min(int(n), n_samples)
    if n_samples < 2:
        return X[:n]

    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]])
    ridge = np.trace(cov) / cov.shape[0]
    ridge = 1e-6 * ridge if np.isfinite(ridge) and ridge > 0 else 1e-6
    cov = cov + ridge * np.eye(cov.shape[0])
    VI = np.linalg.pinv(cov)

    # Score each record against the mean of the remaining points, using a
    # shared precision matrix so the Mahalanobis metric is stable.
    mean_rest = (n_samples * mean - X) / (n_samples - 1)
    distances = np.array([
        mahalanobis(X[i], mean_rest[i], VI) for i in range(n_samples)
    ])

    top_idx = np.argpartition(distances, -n)[-n:]
    top_idx = top_idx[np.argsort(distances[top_idx])[::-1]]
    return top_idx

def create_privacy_flawed_dataset(synthetic_data, landmarks, leak_scale, proportion, unique_target_records):
    """
    Replace a random fraction of synthetic records with privacy-leaked samples.

    Each unique target is spliced into a host synthetic series via
    `inject_segment_leak`. Those leaked traces then overwrite a random
    `proportion` of the synthetic set (sampled with replacement if there are
    fewer leaked traces than slots). Landmarks are remapped with the same
    indices: if synthetic record j is replaced by flawed sample i, landmark j
    is replaced by landmark i.
    """
    n_unique = len(unique_target_records)
    privacy_flawed_samples = np.stack([
        inject_segment_leak(unique_target_records[i], synthetic_data[i], leak_scale)
        for i in range(n_unique)
    ])

    flawed_dataset = np.array(synthetic_data, copy=True)
    flawed_landmarks = np.array(landmarks, copy=True)
    n_synth = flawed_dataset.shape[0]
    n_replace = int(round(float(proportion) * n_synth))
    n_replace = min(max(n_replace, 0), n_synth)
    if n_replace == 0:
        return flawed_dataset, flawed_landmarks

    replace_idx = np.random.choice(n_synth, size=n_replace, replace=False)
    fill_idx = np.random.choice(n_unique, size=n_replace, replace=True)
    flawed_dataset[replace_idx] = privacy_flawed_samples[fill_idx]
    flawed_landmarks[replace_idx] = flawed_landmarks[fill_idx]
    return flawed_dataset, flawed_landmarks

def distribution_plotting(data, name, save_path):
    plt.figure(figsize=(7, 6))
    plt.hist(data, bins=50, label=name)
    plt.xlabel('Score', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title(f'{name}', fontsize=13, weight='bold', pad=15)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path + f'{name}.png', dpi=300)
    plt.close()

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
    sr = 100
    domain_range = (0, 1)
    n_components = 10

    np.random.seed(42)

    save_path = "images/privacy_val/"
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)

    ### Find Highly Unique Sample from Real Aligned Data
    with open(f"data/validation/real_data.pkl", "rb") as f:
        real_data = pickle.load(f)
    with open(f"data/validation/real_fd.pkl", "rb") as f:
        real_fd = pickle.load(f)
    n_sample, n_timepoints, n_channel = real_fd.data_matrix.shape
    n_basis = int(n_timepoints / 2)
    lambda_ = basis_smoothing_hyperparameter_tuning(real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
    real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(real_fd_smooth, n_components)
    unique_record_idx = find_highly_unique_target_records(real_scores, n=5)
    
    ### Get Reference Data
    with open(f"data/validation/substitute_data.pkl", "rb") as f:
        reference_data = pickle.load(f)
    with open(f"data/validation/substitute_landmarks.pkl", "rb") as f:
        reference_landmarks = pickle.load(f)
    with open(f"data/validation/substitute_fd.pkl", "rb") as f:
        holdout_fd = pickle.load(f)
    n_sample, n_timepoints, n_channel = holdout_fd.data_matrix.shape
    n_basis = int(n_timepoints / 2)

    ## Feature Extraction from Reference Data
    # Baseline Transformations: PCA, FFT, Wavelet
    reference_pca_scores, reference_pca_model = pca(reference_data)
    reference_fft_scores, reference_fft_basis = fft(reference_data, k=10)
    reference_wavelet_scores, reference_wavelet_basis = wavelet(reference_data, [(22.5, 45.0, (11.25, 22.5), (5.6, 11.25), (2.8, 5.6))])

    # FPCA + Diffusion Map
    lambda_ = basis_smoothing_hyperparameter_tuning(holdout_fd, n_basis, domain_range)
    holdout_fd_smooth, _, _, _ = basis_smoothing_with_lambda(holdout_fd, lambda_, n_basis, domain_range)
    holdout_mean, holdout_components, holdout_scores, holdout_var_ratio, holdout_fpca_ = fpca_with_param(holdout_fd_smooth, n_components)

    holdout_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(holdout_scores)
    holdout_dmap_evals = holdout_dmap.evals_
    holdout_dmap_embedding = holdout_dmap.transform(holdout_scores)

    ## Feature Extraction from Real Data
    # Baseline Transformations: PCA, FFT, Wavelet
    real_pca_scores = pca_transform(real_data, reference_pca_model)
    real_fft_scores = fft_transform(real_data, reference_fft_basis)
    real_wavelet_scores = wavelet_transform(real_data, reference_wavelet_basis)

    # FPCA + Diffusion Map
    real_fpca_scores = holdout_fpca_.transform(real_fd)
    real_dmap_embedding = holdout_dmap.transform(real_fpca_scores)

    ### Create a Good Synthetic Dataset from Reference (Add minor Gaussian noise to reference data)
    active = np.abs(reference_data) > 1e-6
    data_std = np.std(reference_data[active]) if np.any(active) else np.std(reference_data)
    noise = np.random.normal(loc=0.0, scale=data_std * 0.05, size=reference_data.shape)
    synthetic_data = reference_data + noise
    synthetic_fd = align_ecg(synthetic_data, reference_landmarks)

    ## Feature Extraction from Synthetic Data
    # Baseline Transformations: PCA, FFT, Wavelet
    synthetic_pca_scores = pca_transform(synthetic_data, reference_pca_model)
    synthetic_fft_scores = fft_transform(synthetic_data, reference_fft_basis)
    synthetic_wavelet_scores = wavelet_transform(synthetic_data, reference_wavelet_basis)

    # FPCA + Diffusion Map
    synthetic_fpca_scores = holdout_fpca_.transform(synthetic_fd)
    synthetic_dmap_embedding = holdout_dmap.transform(synthetic_fpca_scores)

    ### Single Sample Attack Percentile Score
    leak_scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    attack_percentile_results = {}

    for i, leak_scale in enumerate(leak_scales):
        attack_percentile_results[leak_scale] = {}

        # Create a privacy flawed sample
        flaw_sample = inject_segment_leak(real_data[unique_record_idx[0]], reference_data[0], leak_scale)
        flaw_fd = align_ecg(flaw_sample, reference_landmarks)

        ### Transformations
        # Baseline Transformations: PCA, FFT, Wavelet
        flaw_pca_scores = pca_transform(flaw_sample, reference_pca_model)
        flaw_fft_scores = fft_transform(flaw_sample, reference_fft_basis)
        flaw_wavelet_scores = wavelet_transform(flaw_sample, reference_wavelet_basis)

        # FPCA + Diffusion Map
        flaw_fpca_scores = holdout_fpca_.transform(flaw_fd)
        flaw_dmap_embedding = holdout_dmap.transform(flaw_fpca_scores)

        ### DCR Attack Percentile Score
        # Baseline: Raw Time-Series
        attack_percentile_results[leak_scale]['raw_dcr_attack_percentile'] = dcr_attack_percentile(flaw_sample, reference_data, synthetic_data)
        attack_percentile_results[leak_scale]['raw_nndr_attack_percentile'] = nndr_attack_percentile(flaw_sample, reference_data, synthetic_data)
        attack_percentile_results[leak_scale]['raw_domias_attack_percentile'] = domias_attack_percentile(flaw_sample, reference_data, synthetic_data)

        # PCA
        attack_percentile_results[leak_scale]['pca_dcr_attack_percentile'] = dcr_attack_percentile(flaw_pca_scores, reference_pca_scores, synthetic_pca_scores)
        attack_percentile_results[leak_scale]['pca_nndr_attack_percentile'] = nndr_attack_percentile(flaw_pca_scores, reference_pca_scores, synthetic_pca_scores)
        attack_percentile_results[leak_scale]['pca_domias_attack_percentile'] = domias_attack_percentile(flaw_pca_scores, reference_pca_scores, synthetic_pca_scores)

        # FFT
        attack_percentile_results[leak_scale]['fft_dcr_attack_percentile'] = dcr_attack_percentile(flaw_fft_scores, reference_fft_scores, synthetic_fft_scores)
        attack_percentile_results[leak_scale]['fft_nndr_attack_percentile'] = nndr_attack_percentile(flaw_fft_scores, reference_fft_scores, synthetic_fft_scores)
        attack_percentile_results[leak_scale]['fft_domias_attack_percentile'] = domias_attack_percentile(flaw_fft_scores, reference_fft_scores, synthetic_fft_scores)

        # Wavelet
        attack_percentile_results[leak_scale]['wavelet_dcr_attack_percentile'] = dcr_attack_percentile(flaw_wavelet_scores, reference_wavelet_scores, synthetic_wavelet_scores)
        attack_percentile_results[leak_scale]['wavelet_nndr_attack_percentile'] = nndr_attack_percentile(flaw_wavelet_scores, reference_wavelet_scores, synthetic_wavelet_scores)
        attack_percentile_results[leak_scale]['wavelet_domias_attack_percentile'] = domias_attack_percentile(flaw_wavelet_scores, reference_wavelet_scores, synthetic_wavelet_scores)

        # FPCA
        attack_percentile_results[leak_scale]['fpc_dcr_attack_percentile'] = dcr_attack_percentile(flaw_fpca_scores, holdout_scores, synthetic_fpca_scores)
        attack_percentile_results[leak_scale]['fpc_nndr_attack_percentile'] = nndr_attack_percentile(flaw_fpca_scores, holdout_scores, synthetic_fpca_scores)
        attack_percentile_results[leak_scale]['fpc_domias_attack_percentile'] = domias_attack_percentile(flaw_fpca_scores, holdout_scores, synthetic_fpca_scores)

        # Diffusion Map
        attack_percentile_results[leak_scale]['dmap_dcr_attack_percentile'] = dcr_attack_percentile(flaw_dmap_embedding, holdout_dmap_embedding, synthetic_dmap_embedding)
        attack_percentile_results[leak_scale]['dmap_nndr_attack_percentile'] = nndr_attack_percentile(flaw_dmap_embedding, holdout_dmap_embedding, synthetic_dmap_embedding)
        attack_percentile_results[leak_scale]['dmap_domias_attack_percentile'] = domias_attack_percentile(flaw_dmap_embedding, holdout_dmap_embedding, synthetic_dmap_embedding)

    ### Save Attack Percentile Results
    with open(f"images/privacy_val/privacy_val_attack_percentile_results.json", "w") as f:
        json.dump(attack_percentile_results, f)

    ### Distribution Wise MIA
    unique_target_records = real_data[unique_record_idx]
    proportions = [0.1, 0.2, 0.3, 0.4, 0.5]
    proportion = 0.1

    raw_dcr_fprs, raw_dcr_tprs, raw_dcr_roc_aucs = [], [], []
    raw_nndr_fprs, raw_nndr_tprs, raw_nndr_roc_aucs = [], [], []
    raw_domias_fprs, raw_domias_tprs, raw_domias_roc_aucs = [], [], []
    pca_dcr_fprs, pca_dcr_tprs, pca_dcr_roc_aucs = [], [], []
    pca_nndr_fprs, pca_nndr_tprs, pca_nndr_roc_aucs = [], [], []
    pca_domias_fprs, pca_domias_tprs, pca_domias_roc_aucs = [], [], []
    fft_dcr_fprs, fft_dcr_tprs, fft_dcr_roc_aucs = [], [], []
    fft_nndr_fprs, fft_nndr_tprs, fft_nndr_roc_aucs = [], [], []
    fft_domias_fprs, fft_domias_tprs, fft_domias_roc_aucs = [], [], []
    wavelet_dcr_fprs, wavelet_dcr_tprs, wavelet_dcr_roc_aucs = [], [], []
    wavelet_nndr_fprs, wavelet_nndr_tprs, wavelet_nndr_roc_aucs = [], [], []
    wavelet_domias_fprs, wavelet_domias_tprs, wavelet_domias_roc_aucs = [], [], []
    fpc_dcr_fprs, fpc_dcr_tprs, fpc_dcr_roc_aucs = [], [], []
    fpc_nndr_fprs, fpc_nndr_tprs, fpc_nndr_roc_aucs = [], [], []
    fpc_domias_fprs, fpc_domias_tprs, fpc_domias_roc_aucs = [], [], []
    dmap_dcr_fprs, dmap_dcr_tprs, dmap_dcr_roc_aucs = [], [], []
    dmap_nndr_fprs, dmap_nndr_tprs, dmap_nndr_roc_aucs = [], [], []
    dmap_domias_fprs, dmap_domias_tprs, dmap_domias_roc_aucs = [], [], []
    for leak_scale in leak_scales:

        flaw_dataset, flaw_landmarks = create_privacy_flawed_dataset(synthetic_data, reference_landmarks, leak_scale, proportion, unique_target_records)
        flaw_fd = align_ecg(flaw_dataset, flaw_landmarks)

        ### Transformations
        # Baseline Transformations: PCA, FFT, Wavelet
        flaw_pca_scores = pca_transform(flaw_dataset, reference_pca_model)
        flaw_fft_scores = fft_transform(flaw_dataset, reference_fft_basis)
        flaw_wavelet_scores = wavelet_transform(flaw_dataset, reference_wavelet_basis)

        # FPCA + Diffusion Map
        flaw_fpca_scores = holdout_fpca_.transform(flaw_fd)
        flaw_dmap_embedding = holdout_dmap.transform(flaw_fpca_scores)

        ### MIA
        # Baseline: Raw Time-Series
        raw_dcr_fpr, raw_dcr_tpr, raw_dcr_roc_auc = dcr_mia(real_data, reference_data, flaw_dataset)
        raw_nndr_fpr, raw_nndr_tpr, raw_nndr_roc_auc = nndr_mia(real_data, reference_data, flaw_dataset)
        raw_domias_fpr, raw_domias_tpr, raw_domias_roc_auc = domias_mia(real_data, reference_data, flaw_dataset)
        raw_dcr_fprs.append(raw_dcr_fpr)
        raw_dcr_tprs.append(raw_dcr_tpr)
        raw_dcr_roc_aucs.append(raw_dcr_roc_auc)
        raw_nndr_fprs.append(raw_nndr_fpr)
        raw_nndr_tprs.append(raw_nndr_tpr)
        raw_nndr_roc_aucs.append(raw_nndr_roc_auc)
        raw_domias_fprs.append(raw_domias_fpr)
        raw_domias_tprs.append(raw_domias_tpr)
        raw_domias_roc_aucs.append(raw_domias_roc_auc)

        # PCA
        pca_dcr_fpr, pca_dcr_tpr, pca_dcr_roc_auc = dcr_mia(real_pca_scores, reference_pca_scores, flaw_pca_scores)
        pca_nndr_fpr, pca_nndr_tpr, pca_nndr_roc_auc = nndr_mia(real_pca_scores, reference_pca_scores, flaw_pca_scores)
        pca_domias_fpr, pca_domias_tpr, pca_domias_roc_auc = domias_mia(real_pca_scores, reference_pca_scores, flaw_pca_scores)
        pca_dcr_fprs.append(pca_dcr_fpr)
        pca_dcr_tprs.append(pca_dcr_tpr)
        pca_dcr_roc_aucs.append(pca_dcr_roc_auc)
        pca_nndr_fprs.append(pca_nndr_fpr)
        pca_nndr_tprs.append(pca_nndr_tpr)
        pca_nndr_roc_aucs.append(pca_nndr_roc_auc)
        pca_domias_fprs.append(pca_domias_fpr)
        pca_domias_tprs.append(pca_domias_tpr)
        pca_domias_roc_aucs.append(pca_domias_roc_auc)

        # FFT
        fft_dcr_fpr, fft_dcr_tpr, fft_dcr_roc_auc = dcr_mia(real_fft_scores, reference_fft_scores, flaw_fft_scores)
        fft_nndr_fpr, fft_nndr_tpr, fft_nndr_roc_auc = nndr_mia(real_fft_scores, reference_fft_scores, flaw_fft_scores)
        fft_domias_fpr, fft_domias_tpr, fft_domias_roc_auc = domias_mia(real_fft_scores, reference_fft_scores, flaw_fft_scores)
        fft_dcr_fprs.append(fft_dcr_fpr)
        fft_dcr_tprs.append(fft_dcr_tpr)
        fft_dcr_roc_aucs.append(fft_dcr_roc_auc)
        fft_nndr_fprs.append(fft_nndr_fpr)
        fft_nndr_tprs.append(fft_nndr_tpr)
        fft_nndr_roc_aucs.append(fft_nndr_roc_auc)
        fft_domias_fprs.append(fft_domias_fpr)
        fft_domias_tprs.append(fft_domias_tpr)
        fft_domias_roc_aucs.append(fft_domias_roc_auc)

        # Wavelet
        wavelet_dcr_fpr, wavelet_dcr_tpr, wavelet_dcr_roc_auc = dcr_mia(real_wavelet_scores, reference_wavelet_scores, flaw_wavelet_scores)
        wavelet_nndr_fpr, wavelet_nndr_tpr, wavelet_nndr_roc_auc = nndr_mia(real_wavelet_scores, reference_wavelet_scores, flaw_wavelet_scores)
        wavelet_domias_fpr, wavelet_domias_tpr, wavelet_domias_roc_auc = domias_mia(real_wavelet_scores, reference_wavelet_scores, flaw_wavelet_scores)
        wavelet_dcr_fprs.append(wavelet_dcr_fpr)
        wavelet_dcr_tprs.append(wavelet_dcr_tpr)
        wavelet_dcr_roc_aucs.append(wavelet_dcr_roc_auc)
        wavelet_nndr_fprs.append(wavelet_nndr_fpr)
        wavelet_nndr_tprs.append(wavelet_nndr_tpr)
        wavelet_nndr_roc_aucs.append(wavelet_nndr_roc_auc)
        wavelet_domias_fprs.append(wavelet_domias_fpr)
        wavelet_domias_tprs.append(wavelet_domias_tpr)
        wavelet_domias_roc_aucs.append(wavelet_domias_roc_auc)

        # FPCA
        fpc_dcr_fpr, fpc_dcr_tpr, fpc_dcr_roc_auc = dcr_mia(real_fpca_scores, holdout_scores, flaw_fpca_scores)
        fpc_nndr_fpr, fpc_nndr_tpr, fpc_nndr_roc_auc = nndr_mia(real_fpca_scores, holdout_scores, flaw_fpca_scores)
        fpc_domias_fpr, fpc_domias_tpr, fpc_domias_roc_auc = domias_mia(real_fpca_scores, holdout_scores, flaw_fpca_scores)
        fpc_dcr_fprs.append(fpc_dcr_fpr)
        fpc_dcr_tprs.append(fpc_dcr_tpr)
        fpc_dcr_roc_aucs.append(fpc_dcr_roc_auc)
        fpc_nndr_fprs.append(fpc_nndr_fpr)
        fpc_nndr_tprs.append(fpc_nndr_tpr)
        fpc_nndr_roc_aucs.append(fpc_nndr_roc_auc)
        fpc_domias_fprs.append(fpc_domias_fpr)
        fpc_domias_tprs.append(fpc_domias_tpr)
        fpc_domias_roc_aucs.append(fpc_domias_roc_auc)

        # Diffusion Map
        dmap_dcr_fpr, dmap_dcr_tpr, dmap_dcr_roc_auc = dcr_mia(real_dmap_embedding, holdout_dmap_embedding, flaw_dmap_embedding)
        dmap_nndr_fpr, dmap_nndr_tpr, dmap_nndr_roc_auc = nndr_mia(real_dmap_embedding, holdout_dmap_embedding, flaw_dmap_embedding)
        dmap_domias_fpr, dmap_domias_tpr, dmap_domias_roc_auc = domias_mia(real_dmap_embedding, holdout_dmap_embedding, flaw_dmap_embedding)
        dmap_dcr_fprs.append(dmap_dcr_fpr)
        dmap_dcr_tprs.append(dmap_dcr_tpr)
        dmap_dcr_roc_aucs.append(dmap_dcr_roc_auc)
        dmap_nndr_fprs.append(dmap_nndr_fpr)
        dmap_nndr_tprs.append(dmap_nndr_tpr)
        dmap_nndr_roc_aucs.append(dmap_nndr_roc_auc)
        dmap_domias_fprs.append(dmap_domias_fpr)
        dmap_domias_tprs.append(dmap_domias_tpr)
        dmap_domias_roc_aucs.append(dmap_domias_roc_auc)


    plot_roc_curve(raw_dcr_fprs, raw_dcr_tprs, raw_dcr_roc_aucs, leak_scales, f'DCR Baseline', save_path)
    plot_roc_curve(raw_nndr_fprs, raw_nndr_tprs, raw_nndr_roc_aucs, leak_scales, f'NNDR Baseline', save_path)
    plot_roc_curve(raw_domias_fprs, raw_domias_tprs, raw_domias_roc_aucs, leak_scales, f'DOMIAS Baseline', save_path)
    plot_roc_curve(pca_dcr_fprs, pca_dcr_tprs, pca_dcr_roc_aucs, leak_scales, f'PCA DCR', save_path)
    plot_roc_curve(pca_nndr_fprs, pca_nndr_tprs, pca_nndr_roc_aucs, leak_scales, f'PCA NNDR', save_path)
    plot_roc_curve(pca_domias_fprs, pca_domias_tprs, pca_domias_roc_aucs, leak_scales, f'PCA DOMIAS', save_path)
    plot_roc_curve(fft_dcr_fprs, fft_dcr_tprs, fft_dcr_roc_aucs, leak_scales, f'FFT DCR', save_path)
    plot_roc_curve(fft_nndr_fprs, fft_nndr_tprs, fft_nndr_roc_aucs, leak_scales, f'FFT NNDR', save_path)
    plot_roc_curve(fft_domias_fprs, fft_domias_tprs, fft_domias_roc_aucs, leak_scales, f'FFT DOMIAS', save_path)
    plot_roc_curve(wavelet_dcr_fprs, wavelet_dcr_tprs, wavelet_dcr_roc_aucs, leak_scales, f'Wavelet DCR', save_path)
    plot_roc_curve(wavelet_nndr_fprs, wavelet_nndr_tprs, wavelet_nndr_roc_aucs, leak_scales, f'Wavelet NNDR', save_path)
    plot_roc_curve(wavelet_domias_fprs, wavelet_domias_tprs, wavelet_domias_roc_aucs, leak_scales, f'Wavelet DOMIAS', save_path)
    plot_roc_curve(fpc_dcr_fprs, fpc_dcr_tprs, fpc_dcr_roc_aucs, leak_scales, f'FPC DCR', save_path)
    plot_roc_curve(fpc_nndr_fprs, fpc_nndr_tprs, fpc_nndr_roc_aucs, leak_scales, f'FPC NNDR', save_path)
    plot_roc_curve(fpc_domias_fprs, fpc_domias_tprs, fpc_domias_roc_aucs, leak_scales, f'FPC DOMIAS', save_path)
    plot_roc_curve(dmap_dcr_fprs, dmap_dcr_tprs, dmap_dcr_roc_aucs, leak_scales, f'DMap DCR', save_path)
    plot_roc_curve(dmap_nndr_fprs, dmap_nndr_tprs, dmap_nndr_roc_aucs, leak_scales, f'DMap NNDR', save_path)
    plot_roc_curve(dmap_domias_fprs, dmap_domias_tprs, dmap_domias_roc_aucs, leak_scales, f'DMap DOMIAS', save_path)
import json
import pickle
import numpy as np
from pathlib import Path
from scenario_engineering.dataset_creation import get_distributional_scenarios, get_morphology_scenarios, get_temporal_scenarios
from transformation.baseline.pca import *
from transformation.baseline.fft import *
from transformation.baseline.wavelet import *
from metrics.fidelity import *

if __name__ == "__main__":
    diagnostic = "NORM"
    lead = 1
    sr = 100
    n_components = 10
    domain_range = (0, 1)

    np.random.seed(42)

    ### Get Real Unaligned and Aligned Data
    with open(f"data/validation/real_data.pkl", "rb") as f:
        real_data = pickle.load(f)

    # Baseline Transformations on Unaligned Data: PCA, FFT, Wavelet
    real_unaligned_pca_scores, real_unaligned_pca_model = pca(real_data)
    real_unaligned_fft_scores, real_unaligned_fft_basis = fft(real_data, k=10)
    real_unaligned_wavelet_scores, real_unaligned_wavelet_basis = wavelet(real_data, [(22.5, 45.0, (11.25, 22.5), (5.6, 11.25), (2.8, 5.6))])

    # # Baseline Transformations on Aligned Data: PCA, FFT, Wavelet
    # real_aligned_pca_scores, real_aligned_pca_model = pca(real_fd.data_matrix.squeeze())
    # real_aligned_fft_scores, real_aligned_fft_basis = fft(real_fd.data_matrix.squeeze(), k=10)
    # real_aligned_wavelet_scores, real_aligned_wavelet_basis = wavelet(real_fd.data_matrix.squeeze(), [(22.5, 45.0, (11.25, 22.5), (5.6, 11.25), (2.8, 5.6))])

    scenarios = get_distributional_scenarios() + get_morphology_scenarios() + get_temporal_scenarios()
    result_tracking = {}
    for scenario in scenarios:

        with open(f"data/validation/morphology/{scenario}_dataset.pkl", "rb") as f:
            datasets = pickle.load(f)
        
        result_tracking[scenario] = {}

        for key, (flaw_data, flaw_fd) in datasets.items():
            #### ------------ Transformations ------------ ####
            # Baseline Transformations on Unaligned Data: PCA, FFT, Wavelet
            flaw_unaligned_pca_scores = pca_transform(flaw_data, real_unaligned_pca_model)
            flaw_unaligned_fft_scores = fft_transform(flaw_data, real_unaligned_fft_basis)
            flaw_unaligned_wavelet_scores = wavelet_transform(flaw_data, real_unaligned_wavelet_basis)

            # # Baseline Transformations on Aligned Data: PCA, FFT, Wavelet
            # flaw_aligned_pca_scores = pca_transform(flaw_fd.data_matrix.squeeze(), real_aligned_pca_model)
            # flaw_aligned_fft_scores = fft_transform(flaw_fd.data_matrix.squeeze(), real_aligned_fft_basis)
            # flaw_aligned_wavelet_scores = wavelet_transform(flaw_fd.data_matrix.squeeze(), real_aligned_wavelet_basis)

            #### ------------ Evaluation ------------ ####
            ## Baseline: Raw unaligned Data
            raw_unaligned_data_wasserstein_score = wasserstein(real_data, flaw_data)
            raw_unlaligned_data_mahalanobis_score = sample_wise_mahalanobis(real_data, flaw_data)
            raw_unaligned_data_autocorrelation_score = autocorrelation_score(real_data, flaw_data)
            raw_unaligned_data_dtw_score = dtw_score(real_data, flaw_data)
            raw_unaligned_precision, raw_unaligned_recall = precision_recall(real_data, flaw_data)

            # # Baseline: Raw aligned Data
            # raw_aligned_precision, raw_aligned_recall = precision_recall(real_fd.data_matrix.squeeze(), flaw_fd.data_matrix.squeeze())

            ## Baseline Transformation on Unaligned Data: PCA, FFT, Wavelet
            unaligned_pca_wasserstein_score = wasserstein(real_unaligned_pca_scores, flaw_unaligned_pca_scores)
            unaligned_fft_wasserstein_score = wasserstein(real_unaligned_fft_scores, flaw_unaligned_fft_scores)
            unaligned_wavelet_wasserstein_score = wasserstein(real_unaligned_wavelet_scores, flaw_unaligned_wavelet_scores)
            unaligned_pca_mahalanobis_score = sample_wise_mahalanobis(real_unaligned_pca_scores, flaw_unaligned_pca_scores)
            unaligned_fft_mahalanobis_score = sample_wise_mahalanobis(real_unaligned_fft_scores, flaw_unaligned_fft_scores)
            unaligned_wavelet_mahalanobis_score = sample_wise_mahalanobis(real_unaligned_wavelet_scores, flaw_unaligned_wavelet_scores)
            unaligned_pca_precision, unaligned_pca_recall = precision_recall(real_unaligned_pca_scores, flaw_unaligned_pca_scores)
            unaligned_fft_precision, unaligned_fft_recall = precision_recall(real_unaligned_fft_scores, flaw_unaligned_fft_scores)
            unaligned_wavelet_precision, unaligned_wavelet_recall = precision_recall(real_unaligned_wavelet_scores, flaw_unaligned_wavelet_scores)

            # ## Baseline Transformation on Aligned Data: PCA, FFT, Wavelet
            # aligned_pca_precision, aligned_pca_recall = precision_recall(real_aligned_pca_scores, flaw_aligned_pca_scores)
            # aligned_fft_precision, aligned_fft_recall = precision_recall(real_aligned_fft_scores, flaw_aligned_fft_scores)
            # aligned_wavelet_precision, aligned_wavelet_recall = precision_recall(real_aligned_wavelet_scores, flaw_aligned_wavelet_scores)

            #### ------------ Result Display ------------ ####
            result_tracking[scenario][key] = {}

            # Raw Unaligned Data: Wasserstein Distance, Mahalanobis Distance
            result_tracking[scenario][key]['raw_unaligned_data_wasserstein_score'] = raw_unaligned_data_wasserstein_score
            result_tracking[scenario][key]['raw_unlaligned_data_mahalanobis_score'] = raw_unlaligned_data_mahalanobis_score

            # Raw Unaligned Data: Autocorrelation Score, DTW Score
            result_tracking[scenario][key]['raw_unaligned_data_autocorrelation_score'] = raw_unaligned_data_autocorrelation_score
            result_tracking[scenario][key]['raw_unaligned_data_dtw_score'] = raw_unaligned_data_dtw_score
            
            # Raw Unaligned Data: Precision, Recall
            result_tracking[scenario][key]['raw_unaligned_data_precision'] = raw_unaligned_precision
            result_tracking[scenario][key]['raw_unaligned_data_recall'] = raw_unaligned_recall
            
            # # Raw Aligned Data: Precision, Recall
            # result_tracking[scenario][key]['raw_aligned_data_precision'] = raw_aligned_precision
            # result_tracking[scenario][key]['raw_aligned_data_recall'] = raw_aligned_recall

            # Baseline Transformations on Unaligned Data: PCA, FFT, Wavelet: Wasserstein Distance, Mahalanobis Distance
            result_tracking[scenario][key]['unaligned_pca_wasserstein_score'] = unaligned_pca_wasserstein_score
            result_tracking[scenario][key]['unaligned_fft_wasserstein_score'] = unaligned_fft_wasserstein_score
            result_tracking[scenario][key]['unaligned_wavelet_wasserstein_score'] = unaligned_wavelet_wasserstein_score
            result_tracking[scenario][key]['unaligned_pca_mahalanobis_score'] = unaligned_pca_mahalanobis_score
            result_tracking[scenario][key]['unaligned_fft_mahalanobis_score'] = unaligned_fft_mahalanobis_score
            result_tracking[scenario][key]['unaligned_wavelet_mahalanobis_score'] = unaligned_wavelet_mahalanobis_score

            # Baseline Transformations on Unaligned Data: PCA, FFT, Wavelet: Precision, Recall
            result_tracking[scenario][key]['unaligned_pca_precision'] = unaligned_pca_precision
            result_tracking[scenario][key]['unaligned_pca_recall'] = unaligned_pca_recall
            result_tracking[scenario][key]['unaligned_fft_precision'] = unaligned_fft_precision
            result_tracking[scenario][key]['unaligned_fft_recall'] = unaligned_fft_recall
            result_tracking[scenario][key]['unaligned_wavelet_precision'] = unaligned_wavelet_precision
            result_tracking[scenario][key]['unaligned_wavelet_recall'] = unaligned_wavelet_recall
    
    # SaveResult Tracking
    with open(f"images/fidelity_val/fidelity_val_baseline_result.json", "w") as f:
        json.dump(result_tracking, f)
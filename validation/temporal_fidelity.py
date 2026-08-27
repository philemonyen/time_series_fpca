import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess.fpca_preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from transformation.fda.fpca import fpca_hyperparameter_tuning, fpca_with_param
from transformation.nonlinear.diffusion_map import DenseDiffusionMap
from transformation.nonlinear.umap import tune_umap
from transformation.baseline.fft import *
from transformation.baseline.pca import *
from transformation.baseline.wavelet import *
from metrics.fidelity import *
from scenario_engineering.dataset_creation import get_temporal_scenarios

if __name__ == "__main__":
    ## ------------ Data Preparation ------------ ##
    diagnostic = "NORM"
    lead = 1
    sr = 100
    n_components = 10
    domain_range = (0, 1)
    np.random.seed(42)

    # Get Real Data and Segments
    with open(f"data/validation/real_data.pkl", "rb") as f:
        real_data = pickle.load(f)
    with open(f"data/validation/real_segments.pkl", "rb") as f:
        real_segments, real_landmarks = pickle.load(f)

    # Baseline Transformations: PCA, FFT, Wavelet
    real_unaligned_pca_scores, real_unaligned_pca_model = pca(real_data)
    real_unaligned_fft_scores, real_unaligned_fft_basis = fft(real_data, k=10)
    real_unaligned_wavelet_scores, real_unaligned_wavelet_basis = wavelet(real_data, [(22.5, 45.0, (11.25, 22.5), (5.6, 11.25), (2.8, 5.6))])

    # Extract Warping Functions and Apply FPCA
    aligned_real_segments, real_warping_ = landmark_registration(real_segments, real_landmarks)
    n_basis = int(real_warping_.data_matrix.shape[1] / 2)
    lambda_ = basis_smoothing_hyperparameter_tuning(real_warping_, n_basis, domain_range)
    smoothed_real, _, _, _ = basis_smoothing_with_lambda(real_warping_, lambda_, n_basis, domain_range)
    n_components = fpca_hyperparameter_tuning(smoothed_real)
    real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(smoothed_real, n_components)

    # Plot FPCA outcomes
    real_mean.plot()
    plt.title("Real Warping Mean")
    plt.xlabel("Time")
    plt.savefig(f"images/fidelity_val/temporal/Warping_Real_Mean.png")
    plt.close()
    for i, c in enumerate(real_components):
        c.plot()
        plt.title(f"Real Warping Component {i}")
        plt.xlabel("Time")
        plt.savefig(f"images/fidelity_val/temporal/Warping_Real_Component_{i}.png")
        plt.close()


    scenarios = get_temporal_scenarios()
    result_tracking = {}
    for scenario in scenarios:
        save_path = f"images/fidelity_val/temporal/{scenario}/"
        path=Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(f"data/validation/{scenario}_dataset.pkl", "rb") as f:
            datasets = pickle.load(f)

        scales = []
        result_tracking[scenario] = {}

        for key, (flaw_data, flaw_segments, segment_landmarks) in datasets.items():
            #### ------------ Transformations ------------ ####
            # Baseline Transformations: PCA, FFT, Wavelet
            flaw_unaligned_pca_scores = pca_transform(flaw_data, real_unaligned_pca_model)
            flaw_unaligned_fft_scores = fft_transform(flaw_data, real_unaligned_fft_basis)
            flaw_unaligned_wavelet_scores = wavelet_transform(flaw_data, real_unaligned_wavelet_basis)

            # FPCA
            aligned_flaw, flaw_warping_ = landmark_registration(flaw_segments, segment_landmarks)
            n_basis = int(flaw_warping_.data_matrix.shape[1] / 2)
            lambda_ = basis_smoothing_hyperparameter_tuning(flaw_warping_, n_basis, domain_range)
            smoothed_flaw, _, _, _ = basis_smoothing_with_lambda(flaw_warping_, lambda_, n_basis, domain_range)
            flaw_scores = real_fpca_.transform(smoothed_flaw)

            # Diffusion Map
            real_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(real_scores)
            real_dmap_evals = real_dmap.evals_
            real_dmap_embedding = real_dmap.transform(real_scores)
            flaw_dmap_embedding = real_dmap.transform(flaw_scores)

            # UMAP
            real_umap = tune_umap(real_scores)
            real_umap_embedding = real_umap.transform(real_scores)
            flaw_umap_embedding = real_umap.transform(flaw_scores)

            #### ------------ Evaluation ------------ ####
            # Baseline: Raw Data
            raw_unaligned_data_frechet_score = frechet_score(real_data, flaw_data)
            raw_unaligned_data_wasserstein_score = wasserstein(real_data, flaw_data)
            raw_unaligned_data_mmd_score = mmd(real_data, flaw_data)

            # Baseline: PCA, FFT, Wavelet
            unaligned_pca_frechet_score = frechet_score(real_unaligned_pca_scores, flaw_unaligned_pca_scores)
            unaligned_pca_wasserstein_score = wasserstein(real_unaligned_pca_scores, flaw_unaligned_pca_scores)
            unaligned_pca_mmd_score = mmd(real_unaligned_pca_scores, flaw_unaligned_pca_scores)
            unaligned_fft_frechet_score = frechet_score(real_unaligned_fft_scores, flaw_unaligned_fft_scores)
            unaligned_fft_wasserstein_score = wasserstein(real_unaligned_fft_scores, flaw_unaligned_fft_scores)
            unaligned_fft_mmd_score = mmd(real_unaligned_fft_scores, flaw_unaligned_fft_scores)
            unaligned_wavelet_frechet_score = frechet_score(real_unaligned_wavelet_scores, flaw_unaligned_wavelet_scores)
            unaligned_wavelet_wasserstein_score = wasserstein(real_unaligned_wavelet_scores, flaw_unaligned_wavelet_scores)
            unaligned_wavelet_mmd_score = mmd(real_unaligned_wavelet_scores, flaw_unaligned_wavelet_scores)

            # FPCA: MMD, LMR
            fpca_frechet_score = frechet_score(real_scores, flaw_scores)
            fpca_score_mmd = mmd(real_scores, flaw_scores)
            fpca_wasserstein_score = wasserstein(real_scores, flaw_scores)

            # Diffusion Map: JS Divergence, MMD, Spectral Distance
            dmap_js_divergence = grid_js_divergence(real_dmap_embedding, flaw_dmap_embedding)
            dmap_mmd = mmd(real_dmap_embedding, flaw_dmap_embedding)
            dmap_spectral_distance = spectral_distance(real_dmap_embedding, flaw_dmap_embedding)

            # UMAP: JS Divergence, MMD, discriminator score
            umap_js_divergence = grid_js_divergence(real_umap_embedding, flaw_umap_embedding)
            umap_mmd = mmd(real_umap_embedding, flaw_umap_embedding)

            #### ------------ Result Display ------------ ####
            result_tracking[scenario][key] = {}
            # Raw Unaligned Data: Frechet Score, Wasserstein Score, MMD Score
            result_tracking[scenario][key]['raw_unaligned_data_frechet_score'] = raw_unaligned_data_frechet_score
            result_tracking[scenario][key]['raw_unaligned_data_wasserstein_score'] = raw_unaligned_data_wasserstein_score
            result_tracking[scenario][key]['raw_unaligned_data_mmd_score'] = raw_unaligned_data_mmd_score
            
            # Baseline Transformations on Unaligned Data: PCA, FFT, Wavelet: Frechet Score, Wasserstein Score, MMD Score
            result_tracking[scenario][key]['unaligned_pca_frechet_score'] = unaligned_pca_frechet_score
            result_tracking[scenario][key]['unaligned_pca_wasserstein_score'] = unaligned_pca_wasserstein_score
            result_tracking[scenario][key]['unaligned_pca_mmd_score'] = unaligned_pca_mmd_score
            result_tracking[scenario][key]['unaligned_fft_frechet_score'] = unaligned_fft_frechet_score
            result_tracking[scenario][key]['unaligned_fft_wasserstein_score'] = unaligned_fft_wasserstein_score
            result_tracking[scenario][key]['unaligned_fft_mmd_score'] = unaligned_fft_mmd_score
            result_tracking[scenario][key]['unaligned_wavelet_frechet_score'] = unaligned_wavelet_frechet_score
            result_tracking[scenario][key]['unaligned_wavelet_wasserstein_score'] = unaligned_wavelet_wasserstein_score
            result_tracking[scenario][key]['unaligned_wavelet_mmd_score'] = unaligned_wavelet_mmd_score
            
            # FPCA: MMD, Wasserstein, LMR
            result_tracking[scenario][key]['fpca_score_mmd'] = fpca_score_mmd
            result_tracking[scenario][key]['fpca_wasserstein_score'] = fpca_wasserstein_score

            # Diffusion Map: JS Divergence, MMD, Spectral Distance
            result_tracking[scenario][key]['dmap_js_divergence'] = dmap_js_divergence
            result_tracking[scenario][key]['dmap_mmd'] = dmap_mmd
            result_tracking[scenario][key]['dmap_spectral_distance'] = dmap_spectral_distance

            # UMAP: JS Divergence, MMD, discriminator score
            result_tracking[scenario][key]['umap_js_divergence'] = umap_js_divergence
            result_tracking[scenario][key]['umap_mmd'] = umap_mmd

            plt.scatter(real_umap_embedding[:, 0], real_umap_embedding[:, 1], label="Real")
            plt.scatter(flaw_umap_embedding[:, 0], flaw_umap_embedding[:, 1], label="Flaw")
            plt.title(f"UMAP Embedding: {scenario}, Flaw Scale: {key}")
            plt.legend()
            plt.savefig(save_path + f"UMAP_Embedding_{scenario}_{key}.png")
            plt.close()
    
    # Save Result Tracking
    with open(f"images/fidelity_val/temporal/fidelity_val_fpca_result.json", "w") as f:
        json.dump(result_tracking, f)


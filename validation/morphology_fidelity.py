import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skfda.representation.grid import FDataGrid
from preprocess.fpca_preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda
from transformation.fda.fpca import fpca_with_param
from transformation.nonlinear.diffusion_map import DenseDiffusionMap
from transformation.nonlinear.umap import tune_umap
from scenario_engineering.dataset_creation import get_morphology_scenarios, get_distributional_scenarios
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

    ### Get Real Unaligned Data
    with open(f"data/validation/real_data.pkl", "rb") as f:
        real_data = pickle.load(f)

    # Baseline Transformations on Unaligned Data: PCA, FFT, Wavelet
    real_unaligned_pca_scores, real_unaligned_pca_model = pca(real_data)
    real_unaligned_fft_scores, real_unaligned_fft_basis = fft(real_data, k=10)
    real_unaligned_wavelet_scores, real_unaligned_wavelet_basis = wavelet(real_data, [(22.5, 45.0, (11.25, 22.5), (5.6, 11.25), (2.8, 5.6))])

    ### Get Real Aligned Data
    with open(f"data/validation/real_fd.pkl", "rb") as f:
        real_fd = pickle.load(f)
    n_sample, n_timepoints, n_channel = real_fd.data_matrix.shape
    n_basis = int(n_timepoints / 2)

    # Baseline Transformations on Aligned Data: PCA, FFT, Wavelet
    real_aligned_pca_scores, real_aligned_pca_model = pca(real_fd.data_matrix.squeeze())
    real_aligned_fft_scores, real_aligned_fft_basis = fft(real_fd.data_matrix.squeeze(), k=10)
    real_aligned_wavelet_scores, real_aligned_wavelet_basis = wavelet(real_fd.data_matrix.squeeze(), [(22.5, 45.0, (11.25, 22.5), (5.6, 11.25), (2.8, 5.6))])

    # FPCA on real aligned data
    lambda_ = basis_smoothing_hyperparameter_tuning(real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
    real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(real_fd_smooth, n_components)
    real_fd_grid = real_fpca_.components_.grid_points[0]

    scenarios = get_morphology_scenarios()
    result_tracking = {}
    for scenario in scenarios:
        # Result save path
        save_path = f"images/fidelity_val/morphology/{scenario}/"
        path=Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        with open(f"data/validation/{scenario}_dataset.pkl", "rb") as f:
            datasets = pickle.load(f)
        
        result_tracking[scenario] = {}

        for key, (flaw_data, flaw_fd) in datasets.items():
            #### ------------ Transformations ------------ ####
            # Baseline Transformations on Unaligned Data: PCA, FFT, Wavelet
            flaw_unaligned_pca_scores = pca_transform(flaw_data, real_unaligned_pca_model)
            flaw_unaligned_fft_scores = fft_transform(flaw_data, real_unaligned_fft_basis)
            flaw_unaligned_wavelet_scores = wavelet_transform(flaw_data, real_unaligned_wavelet_basis)

            # Baseline Transformations on Aligned Data: PCA, FFT, Wavelet
            flaw_aligned_pca_scores = pca_transform(flaw_fd.data_matrix.squeeze(), real_aligned_pca_model)
            flaw_aligned_fft_scores = fft_transform(flaw_fd.data_matrix.squeeze(), real_aligned_fft_basis)
            flaw_aligned_wavelet_scores = wavelet_transform(flaw_fd.data_matrix.squeeze(), real_aligned_wavelet_basis)

            # FPCA
            lambda_ = basis_smoothing_hyperparameter_tuning(flaw_fd, n_basis, domain_range)
            flaw_fd_smooth, _, _, _ = basis_smoothing_with_lambda(flaw_fd, lambda_, n_basis, domain_range)
            flaw_data_matrix = flaw_fd_smooth(real_fd_grid)
            flaw_fd_smooth = FDataGrid(data_matrix=flaw_data_matrix, grid_points=real_fd_grid)
            flaw_scores = real_fpca_.transform(flaw_fd_smooth)

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
            ## Baseline: Raw unaligned Data
            raw_unaligned_data_frechet_score = frechet_score(real_data, flaw_data)
            raw_unaligned_data_wasserstein_score = wasserstein(real_data, flaw_data)
            raw_unaligned_data_mmd_score = mmd(real_data, flaw_data)

            ## Baseline: Raw aligned Data
            raw_aligned_data_frechet_score = frechet_score(real_fd.data_matrix.squeeze(), flaw_fd.data_matrix.squeeze())
            raw_aligned_data_wasserstein_score = wasserstein(real_fd.data_matrix.squeeze(), flaw_fd.data_matrix.squeeze())
            raw_aligned_data_mmd_score = mmd(real_fd.data_matrix.squeeze(), flaw_fd.data_matrix.squeeze())

            ## Baseline Transformation on Unaligned Data: PCA, FFT, Wavelet
            unaligned_pca_frechet_score = frechet_score(real_unaligned_pca_scores, flaw_unaligned_pca_scores)
            unaligned_pca_wasserstein_score = wasserstein(real_unaligned_pca_scores, flaw_unaligned_pca_scores)
            unaligned_pca_mmd_score = mmd(real_unaligned_pca_scores, flaw_unaligned_pca_scores)
            unaligned_fft_frechet_score = frechet_score(real_unaligned_fft_scores, flaw_unaligned_fft_scores)
            unaligned_fft_wasserstein_score = wasserstein(real_unaligned_fft_scores, flaw_unaligned_fft_scores)
            unaligned_fft_mmd_score = mmd(real_unaligned_fft_scores, flaw_unaligned_fft_scores)
            unaligned_wavelet_frechet_score = frechet_score(real_unaligned_wavelet_scores, flaw_unaligned_wavelet_scores)
            unaligned_wavelet_wasserstein_score = wasserstein(real_unaligned_wavelet_scores, flaw_unaligned_wavelet_scores)
            unaligned_wavelet_mmd_score = mmd(real_unaligned_wavelet_scores, flaw_unaligned_wavelet_scores)

            ## Baseline Transformation on Aligned Data: PCA, FFT, Wavelet
            aligned_pca_frechet_score = frechet_score(real_aligned_pca_scores, flaw_aligned_pca_scores)
            aligned_pca_wasserstein_score = wasserstein(real_aligned_pca_scores, flaw_aligned_pca_scores)
            aligned_pca_mmd_score = mmd(real_aligned_pca_scores, flaw_aligned_pca_scores)
            aligned_fft_frechet_score = frechet_score(real_aligned_fft_scores, flaw_aligned_fft_scores)
            aligned_fft_wasserstein_score = wasserstein(real_aligned_fft_scores, flaw_aligned_fft_scores)
            aligned_fft_mmd_score = mmd(real_aligned_fft_scores, flaw_aligned_fft_scores)
            aligned_wavelet_frechet_score = frechet_score(real_aligned_wavelet_scores, flaw_aligned_wavelet_scores)
            aligned_wavelet_wasserstein_score = wasserstein(real_aligned_wavelet_scores, flaw_aligned_wavelet_scores)
            aligned_wavelet_mmd_score = mmd(real_aligned_wavelet_scores, flaw_aligned_wavelet_scores)

            ## FPC Score: Frechet Score, Wasserstein Score, MMD Score, LMR
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
            
            # Raw Aligned Data: Frechet Score, Wasserstein Score, MMD Score
            result_tracking[scenario][key]['raw_aligned_data_frechet_score'] = raw_aligned_data_frechet_score
            result_tracking[scenario][key]['raw_aligned_data_wasserstein_score'] = raw_aligned_data_wasserstein_score
            result_tracking[scenario][key]['raw_aligned_data_mmd_score'] = raw_aligned_data_mmd_score
            
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
            
            # Baseline Transformations on Aligned Data: PCA, FFT, Wavelet: Frechet Score, Wasserstein Score, MMD Score
            result_tracking[scenario][key]['aligned_pca_frechet_score'] = aligned_pca_frechet_score
            result_tracking[scenario][key]['aligned_pca_wasserstein_score'] = aligned_pca_wasserstein_score
            result_tracking[scenario][key]['aligned_pca_mmd_score'] = aligned_pca_mmd_score
            result_tracking[scenario][key]['aligned_fft_frechet_score'] = aligned_fft_frechet_score
            result_tracking[scenario][key]['aligned_fft_wasserstein_score'] = aligned_fft_wasserstein_score
            result_tracking[scenario][key]['aligned_fft_mmd_score'] = aligned_fft_mmd_score
            result_tracking[scenario][key]['aligned_wavelet_frechet_score'] = aligned_wavelet_frechet_score
            result_tracking[scenario][key]['aligned_wavelet_wasserstein_score'] = aligned_wavelet_wasserstein_score
            result_tracking[scenario][key]['aligned_wavelet_mmd_score'] = aligned_wavelet_mmd_score
            
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
    with open(f"images/fidelity_val/morphology/fidelity_val_fpca_result.json", "w") as f:
        json.dump(result_tracking, f)


    scenarios = get_distributional_scenarios()
    result_tracking = {}
    for scenario in scenarios:
        # Result save path
        save_path = f"images/fidelity_val/distributional/{scenario}/"
        path=Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        with open(f"data/validation/{scenario}_dataset.pkl", "rb") as f:
            datasets = pickle.load(f)
        
        result_tracking[scenario] = {}

        for key, (flaw_data, flaw_fd) in datasets.items():
            #### ------------ Transformations ------------ ####
            # Baseline Transformations on Unaligned Data: PCA, FFT, Wavelet
            n = pca_n_components(flaw_data)
            flaw_unaligned_pca_scores = pca_transform(flaw_data, real_unaligned_pca_model)
            flaw_unaligned_fft_scores = fft_transform(flaw_data, real_unaligned_fft_basis)
            flaw_unaligned_wavelet_scores = wavelet_transform(flaw_data, real_unaligned_wavelet_basis)

            # Baseline Transformations on Aligned Data: PCA, FFT, Wavelet
            flaw_aligned_pca_scores = pca_transform(flaw_fd.data_matrix.squeeze(), real_aligned_pca_model)
            flaw_aligned_fft_scores = fft_transform(flaw_fd.data_matrix.squeeze(), real_aligned_fft_basis)
            flaw_aligned_wavelet_scores = wavelet_transform(flaw_fd.data_matrix.squeeze(), real_aligned_wavelet_basis)

            # FPCA
            lambda_ = basis_smoothing_hyperparameter_tuning(flaw_fd, n_basis, domain_range)
            flaw_fd_smooth, _, _, _ = basis_smoothing_with_lambda(flaw_fd, lambda_, n_basis, domain_range)
            flaw_data_matrix = flaw_fd_smooth(real_fd_grid)
            flaw_fd_smooth = FDataGrid(data_matrix=flaw_data_matrix, grid_points=real_fd_grid)
            flaw_scores = real_fpca_.transform(flaw_fd_smooth)

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
            ## Baseline: Raw unaligned Data
            raw_unaligned_precision, raw_unaligned_recall = precision_recall(real_data, flaw_data)

            ## Baseline: Raw aligned Data
            raw_aligned_precision, raw_aligned_recall = precision_recall(real_fd.data_matrix.squeeze(), flaw_fd.data_matrix.squeeze())

            ## Baseline Transformation on Unaligned Data: PCA, FFT, Wavelet
            unaligned_pca_precision, unaligned_pca_recall = precision_recall(real_unaligned_pca_scores, flaw_unaligned_pca_scores)
            unaligned_fft_precision, unaligned_fft_recall = precision_recall(real_unaligned_fft_scores, flaw_unaligned_fft_scores)
            unaligned_wavelet_precision, unaligned_wavelet_recall = precision_recall(real_unaligned_wavelet_scores, flaw_unaligned_wavelet_scores)

            ## Baseline Transformation on Aligned Data: PCA, FFT, Wavelet
            aligned_pca_precision, aligned_pca_recall = precision_recall(real_aligned_pca_scores, flaw_aligned_pca_scores)
            aligned_fft_precision, aligned_fft_recall = precision_recall(real_aligned_fft_scores, flaw_aligned_fft_scores)
            aligned_wavelet_precision, aligned_wavelet_recall = precision_recall(real_aligned_wavelet_scores, flaw_aligned_wavelet_scores)
            
            ## FPC Score:
            fpca_precision, fpca_recall = precision_recall(real_scores, flaw_scores)

            # Diffusion Map: JS Divergence, MMD, Spectral Distance
            dmap_precision, dmap_recall = precision_recall(real_dmap_embedding, flaw_dmap_embedding)

            # UMAP: JS Divergence, MMD, discriminator score
            umap_precision, umap_recall = precision_recall(real_umap_embedding, flaw_umap_embedding)

            #### ------------ Result Display ------------ ####
            result_tracking[scenario][key] = {}
            # Raw Unaligned Data: Precision, Recall
            result_tracking[scenario][key]['raw_unaligned_data_precision'] = raw_unaligned_precision
            result_tracking[scenario][key]['raw_unaligned_data_recall'] = raw_unaligned_recall
            
            # Raw Aligned Data: Precision, Recall
            result_tracking[scenario][key]['raw_aligned_data_precision'] = raw_aligned_precision
            result_tracking[scenario][key]['raw_aligned_data_recall'] = raw_aligned_recall
            
            # Baseline Transformations on Unaligned Data: PCA, FFT, Wavelet: Precision, Recall
            result_tracking[scenario][key]['unaligned_pca_precision'] = unaligned_pca_precision
            result_tracking[scenario][key]['unaligned_pca_recall'] = unaligned_pca_recall
            result_tracking[scenario][key]['unaligned_fft_precision'] = unaligned_fft_precision
            result_tracking[scenario][key]['unaligned_fft_recall'] = unaligned_fft_recall
            result_tracking[scenario][key]['unaligned_wavelet_precision'] = unaligned_wavelet_precision
            result_tracking[scenario][key]['unaligned_wavelet_recall'] = unaligned_wavelet_recall
            
            # Baseline Transformations on Aligned Data: PCA, FFT, Wavelet: Precision, Recall
            result_tracking[scenario][key]['aligned_pca_precision'] = aligned_pca_precision
            result_tracking[scenario][key]['aligned_pca_recall'] = aligned_pca_recall
            result_tracking[scenario][key]['aligned_fft_precision'] = aligned_fft_precision
            result_tracking[scenario][key]['aligned_fft_recall'] = aligned_fft_recall
            result_tracking[scenario][key]['aligned_wavelet_precision'] = aligned_wavelet_precision
            result_tracking[scenario][key]['aligned_wavelet_recall'] = aligned_wavelet_recall
            
            # FPCA: Precision, Recall
            result_tracking[scenario][key]['fpca_precision'] = fpca_precision
            result_tracking[scenario][key]['fpca_recall'] = fpca_recall

            # Diffusion Map: Precision, Recall
            result_tracking[scenario][key]['dmap_precision'] = dmap_precision
            result_tracking[scenario][key]['dmap_recall'] = dmap_recall

            # UMAP: Precision, Recall
            result_tracking[scenario][key]['umap_precision'] = umap_precision
            result_tracking[scenario][key]['umap_recall'] = umap_recall
    
    # SaveResult Tracking
    with open(f"images/fidelity_val/distributional/fidelity_val_distributional_result.json", "w") as f:
        json.dump(result_tracking, f)
import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess.fpca_preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda
from transformation.fda.fpca import fpca_with_param
from transformation.nonlinear.diffusion_map import DenseDiffusionMap
from transformation.nonlinear.umap import tune_umap
from metrics.fidelity import *

if __name__ == "__main__":
    ## ------------ Data Preparation ------------ ##
    diagnostic = "NORM"
    lead = 1
    sr = 100
    n_components = 10
    domain_range = (0, 1)

    np.random.seed(42)

    # Get Real Data
    with open(f"data/validation/real_fd.pkl", "rb") as f:
        real_fd = pickle.load(f)
    n_sample, n_timepoints, n_channel = real_fd.data_matrix.shape
    n_basis = int(n_timepoints / 2)

    lambda_ = basis_smoothing_hyperparameter_tuning(real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
    real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(real_fd_smooth, n_components)

    # Create Controlled Flaw Dataset
    scenarios = ["oversmoothing", "memorization", "gaussian_noise", "mode_collapse_vary_modes", "mode_collapse_vary_spike_ratio", "segment_leaking"]
    datasets = {}
    result_tracking = {}
    
    for scenario in scenarios:
        # Result save path
        save_path = f"images/fidelity_val/morphology/{scenario}/"
        path=Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        with open(f"data/validation/{scenario}_dataset.pkl", "rb") as f:
            datasets = pickle.load(f)
        
        lmr = []
        scales = []
        result_tracking[scenario] = {}

        for key, flaw_fd in datasets.items():
            #### ------------ Transformations ------------ ####
            # FPCA
            lambda_ = basis_smoothing_hyperparameter_tuning(flaw_fd, n_basis, domain_range)
            flaw_fd_smooth, _, _, _ = basis_smoothing_with_lambda(flaw_fd, lambda_, n_basis, domain_range)
            flaw_scores = real_fpca_.transform(flaw_fd_smooth)

            # Diffusion Map
            real_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(real_scores)
            real_dmap_evals = real_dmap.evals_
            real_dmap_embedding = real_dmap.transform(real_scores)
            flaw_dmap_embedding = real_dmap.transform(flaw_scores)

            # UMAP
            # Apply UMAP on real and transform synthetic 
            real_umap = tune_umap(real_scores)
            real_umap_embedding = real_umap.transform(real_scores)
            flaw_umap_embedding = real_umap.transform(flaw_scores)

            #### ------------ Evaluation ------------ ####
            # Baseline
            baseline_discriminative_score = raw_data_discriminative_score(real_fd.data_matrix, flaw_fd.data_matrix)
            # baseline_autocorrelation_score = autocorrelation_score(real_fd.data_matrix, flaw_fd.data_matrix)
            # baseline_dtw_score = dtw_score(real_fd.data_matrix, flaw_fd.data_matrix)
            baseline_frechet_score = frechet_score(real_fd.data_matrix, flaw_fd.data_matrix)

            # FPCA: MMD, LMR
            fpca_discriminative_score = feature_discriminative_score(real_scores, flaw_scores)
            fpca_score_mmd = mmd(real_scores, flaw_scores)
            fpca_wasserstein_score = wasserstein(real_scores, flaw_scores)
            fpca_lmr = local_mixing_ratio(real_scores, flaw_scores)
            lmr.append(fpca_lmr)
            scales.append(key)

            # Diffusion Map: JS Divergence, MMD, Spectral Distance
            dmap_js_divergence = grid_js_divergence(real_dmap_embedding, flaw_dmap_embedding)
            dmap_mmd = mmd(real_dmap_embedding, flaw_dmap_embedding)
            dmap_spectral_distance = spectral_distance(real_dmap_embedding, flaw_dmap_embedding)

            # UMAP: JS Divergence, MMD, discriminator score
            umap_js_divergence = grid_js_divergence(real_umap_embedding, flaw_umap_embedding)
            umap_mmd = mmd(real_umap_embedding, flaw_umap_embedding)
            umap_discriminative_score = feature_discriminative_score(real_umap_embedding, flaw_umap_embedding)

            #### ------------ Result Display ------------ ####
            result_tracking[scenario][key] = {}
            # Baseline: Discriminative Score, Frechet Score
            result_tracking[scenario][key]['baseline_discriminative_score'] = baseline_discriminative_score
            result_tracking[scenario][key]['baseline_frechet_score'] = baseline_frechet_score
            # FPCA: MMD, Wasserstein, LMR
            result_tracking[scenario][key]['fpca_score_mmd'] = fpca_score_mmd
            result_tracking[scenario][key]['fpca_wasserstein_score'] = fpca_wasserstein_score
            result_tracking[scenario][key]['fpca_discriminator_score'] = fpca_discriminative_score
            result_tracking[scenario][key]['dmap_js_divergence'] = dmap_js_divergence
            result_tracking[scenario][key]['dmap_mmd'] = dmap_mmd
            result_tracking[scenario][key]['dmap_spectral_distance'] = dmap_spectral_distance
            result_tracking[scenario][key]['umap_js_divergence'] = umap_js_divergence
            result_tracking[scenario][key]['umap_mmd'] = umap_mmd
            result_tracking[scenario][key]['umap_discriminative_score'] = umap_discriminative_score

            plt.scatter(real_umap_embedding[:, 0], real_umap_embedding[:, 1], label="Real")
            plt.scatter(flaw_umap_embedding[:, 0], flaw_umap_embedding[:, 1], label="Flaw")
            plt.title(f"UMAP Embedding: {scenario}, Flaw Scale: {key}")
            plt.legend()
            plt.savefig(save_path + f"UMAP_Embedding_{scenario}_{key}.png")
            plt.close()

        for i in range(len(lmr)):
            plt.plot(lmr[i][2], lmr[i][0], label=scales[i])
            plt.xlabel("Number of Neighbors")
            plt.ylabel("Local Mixing Ratio")
            plt.title(f"FPCA LMR: {scenario}")
            plt.legend()
        plt.savefig(save_path + f"FPCA_LMR_{scenario}.png")
        plt.close()
    
    # Print Result Tracking
    with open(f"images/fidelity_val/morphology/fidelity_val_fpca_result.json", "w") as f:
        json.dump(result_tracking, f)
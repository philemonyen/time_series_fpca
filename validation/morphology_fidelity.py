import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skfda.representation.grid import FDataGrid
from transformation.fda.fpca import fpca_with_param, basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda
from transformation.nonlinear.diffusion_map import DenseDiffusionMap
from transformation.nonlinear.umap import tune_umap
from scenario_engineering.dataset_creation import get_morphology_scenarios, get_temporal_scenarios, get_distributional_scenarios
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
    with open(f"data/validation/real_fd.pkl", "rb") as f:
        real_fd = pickle.load(f)
    n_sample, n_timepoints, n_channel = real_fd.data_matrix.shape
    n_basis = int(n_timepoints / 2)

    # FPCA on real aligned data
    lambda_ = basis_smoothing_hyperparameter_tuning(real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
    real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(real_fd_smooth, n_components)
    real_fd_grid = real_fpca_.components_.grid_points[0]
    real_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(real_scores)
    real_umap = tune_umap(real_scores)

    scenarios = get_morphology_scenarios() + get_temporal_scenarios() + get_distributional_scenarios()
    result_tracking = {}
    for scenario in scenarios:
        # Result save path
        save_path = f"images/fidelity_val/morphology/{scenario}/"
        path=Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        with open(f"data/validation/morphology/{scenario}_dataset.pkl", "rb") as f:
            datasets = pickle.load(f)
        
        result_tracking[scenario] = {}

        for key, (flaw_data, flaw_fd) in datasets.items():
            # FPCA
            lambda_ = basis_smoothing_hyperparameter_tuning(flaw_fd, n_basis, domain_range)
            flaw_fd_smooth, _, _, _ = basis_smoothing_with_lambda(flaw_fd, lambda_, n_basis, domain_range)
            flaw_data_matrix = flaw_fd_smooth(real_fd_grid)
            flaw_fd_smooth = FDataGrid(data_matrix=flaw_data_matrix, grid_points=real_fd_grid)
            flaw_scores = real_fpca_.transform(flaw_fd_smooth)

            # Diffusion Map
            real_dmap_embedding = real_dmap.transform(real_scores)
            flaw_dmap_embedding = real_dmap.transform(flaw_scores)

            # UMAP
            real_umap_embedding = real_umap.transform(real_scores)
            flaw_umap_embedding = real_umap.transform(flaw_scores)

            #### ------------ Evaluation ------------ ####
            ## FPC Score: Wasserstein Distance, Mahalanobis Distance
            fpca_wasserstein_score = wasserstein(real_scores, flaw_scores)
            fpca_mahalanobis_score = sample_wise_mahalanobis(real_scores, flaw_scores)

            # Diffusion Map: JS Divergence, MMD, Spectral Distance
            dmap_js_divergence = grid_js_divergence(real_dmap_embedding, flaw_dmap_embedding)
            dmap_mahalanobis_score = sample_wise_mahalanobis(real_dmap_embedding, flaw_dmap_embedding)

            # UMAP: JS Divergence, Mahalanobis Distance
            umap_js_divergence = grid_js_divergence(real_umap_embedding, flaw_umap_embedding)
            umap_mahalanobis_score = sample_wise_mahalanobis(real_umap_embedding, flaw_umap_embedding)

            #### ------------ Result Display ------------ ####
            result_tracking[scenario][key] = {}
            # FPCA: Wasserstein Distance, Mahalanobis Distance
            result_tracking[scenario][key]['fpca_wasserstein_score'] = fpca_wasserstein_score
            result_tracking[scenario][key]['fpca_mahalanobis_score'] = fpca_mahalanobis_score

            # Diffusion Map: JS Divergence, Mahalanobis Distance
            result_tracking[scenario][key]['dmap_js_divergence'] = dmap_js_divergence
            result_tracking[scenario][key]['dmap_mahalanobis_score'] = dmap_mahalanobis_score

            # UMAP: JS Divergence, Mahalanobis Distance
            result_tracking[scenario][key]['umap_js_divergence'] = umap_js_divergence
            result_tracking[scenario][key]['umap_mahalanobis_score'] = umap_mahalanobis_score

            plt.scatter(real_umap_embedding[:, 0], real_umap_embedding[:, 1], label="Real")
            plt.scatter(flaw_umap_embedding[:, 0], flaw_umap_embedding[:, 1], label="Flaw")
            plt.title(f"UMAP Embedding: {scenario}, Flaw Scale: {key}")
            plt.legend()
            plt.savefig(save_path + f"UMAP_Embedding_{scenario}_{key}.png")
            plt.close()
    
    # Save Result Tracking
    with open(f"images/fidelity_val/fidelity_val_morphology_result.json", "w") as f:
        json.dump(result_tracking, f)
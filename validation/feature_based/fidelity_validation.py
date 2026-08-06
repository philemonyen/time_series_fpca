import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess.ptbxl_preprocess import load_dataset, get_sr, extract_ecg_phase_aligned
from preprocess.fpca_preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from transformation.fda.fpca import fpca_with_param
from transformation.nonlinear.diffusion_map import DenseDiffusionMap
from transformation.nonlinear.umap import tune_umap
from metrics.fidelity import *
from scenario_engineering.dataset_creation import *

if __name__ == "__main__":
    ## ------------ Data Preparation ------------ ##
    diagnostic = "NORM"
    lead = 1
    sr = get_sr()
    n_components = 10
    domain_range = (0, 1)

    # Result save path
    save_path = f"images/fidelity_val/fpca/"
    path=Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Get Real Data
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    aligned_real_fd = extract_ecg_phase_aligned(real_all, sr)
    n_sample, n_timepoints, n_channel = aligned_real_fd.data_matrix.shape
    n_basis = int(n_timepoints / 2)

    real_fd = aligned_real_fd[:n_sample//2]
    substitute_fd = aligned_real_fd[n_sample//2:]

    # Create Controlled Flaw Dataset
    scenarios = ["oversmoothing", "memorization", "gaussian_noise", "mode_collapse_vary_modes", "mode_collapse_vary_spike_ratio", "segment_leaking"]
    datasets = {}
    # Result Tracking
    result_tracking = {}
    for scenario in scenarios:
        with open(save_path + f"{scenario}_dataset.pkl", "rb") as f:
            datasets = pickle.load(f)
        
        lmr = []
        scales = []
        result_tracking[scenario] = {}

        for key, value in datasets.items():
            flaw_fd = value
            #### ------------ Individual FPCA ------------ ####
            # Apply FPCA on Real dataset
            lambda_ = basis_smoothing_hyperparameter_tuning(real_fd, n_basis, domain_range)
            real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
            real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(real_fd_smooth, n_components)

            # Apply FPCA on flaw dataset
            lambda_ = basis_smoothing_hyperparameter_tuning(flaw_fd, n_basis, domain_range)
            flaw_fd_smooth, _, _, _ = basis_smoothing_with_lambda(flaw_fd, lambda_, n_basis, domain_range)

            #### ------------ Shared FPCA ------------ ####
            # Apply Real FPCA on Synthetic
            shared_flaw_scores = real_fpca_.transform(flaw_fd_smooth)

            # Evaluation: MMD, LMR
            fpca_score_mmd = mmd(real_scores, shared_flaw_scores)
            fpca_lmr = local_mixing_ratio(real_scores, shared_flaw_scores)
            lmr.append(fpca_lmr)
            scales.append(key)

            # Apply Diffusion Map on real and synthetic FPC scores separately
            real_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(real_scores)
            real_dmap_evals = real_dmap.evals_
            real_dmap_embedding = real_dmap.transform(real_scores)

            # Apply Diffusion Map on real and transform synthetic 
            flaw_dmap_embedding_shared_real = real_dmap.transform(shared_flaw_scores)

            ## Evaluation: JS Divergence, MMD, PRDC, LMR
            dmap_js_divergence = grid_js_divergence(real_dmap_embedding, flaw_dmap_embedding_shared_real)
            dmap_mmd = mmd(real_dmap_embedding, flaw_dmap_embedding_shared_real)

            # Apply UMAP on real and synthetic FPC scores separately
            real_umap = tune_umap(real_scores)
            real_umap_embedding = real_umap.transform(real_scores)

            # Apply UMAP on real and transform synthetic 
            flaw_umap_embedding_shared_real = real_umap.transform(shared_flaw_scores)

            ## Evaluation: JS Divergence, MMD, PRDC, LMR
            umap_js_divergence = grid_js_divergence(real_umap_embedding, flaw_umap_embedding_shared_real)
            umap_mmd = mmd(real_umap_embedding, flaw_umap_embedding_shared_real)

            #### ------------ Result Display ------------ ####
            result_tracking[scenario][key] = {}

            ## Shared FPCA
            # FPC Score MMD, LMR
            result_tracking[scenario][key]['fpca_score_mmd'] = fpca_score_mmd

            # Shared Diffusion Map: JS Divergence, MMD, PRDC, LMR
            result_tracking[scenario][key]['dmap_js_divergence'] = dmap_js_divergence
            result_tracking[scenario][key]['dmap_mmd'] = dmap_mmd

            # Shared UMAP: JS Divergence, MMD, LMR
            result_tracking[scenario][key]['umap_js_divergence'] = umap_js_divergence
            result_tracking[scenario][key]['umap_mmd'] = umap_mmd

            plt.scatter(real_umap_embedding[:, 0], real_umap_embedding[:, 1], label="Real")
            plt.scatter(flaw_umap_embedding_shared_real[:, 0], flaw_umap_embedding_shared_real[:, 1], label="Flaw")
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
    with open(save_path + f"fidelity_val_fpca_result.txt", "w") as f:
        for scenario in scenarios:
            for key in result_tracking[scenario].keys():
                f.write(f"Scenario: {scenario}, Flaw Scale: {key}\n")
                for key, value in result_tracking[scenario][key].items():
                    f.write(f"    {key}: {value}\n")
                f.write("\n")
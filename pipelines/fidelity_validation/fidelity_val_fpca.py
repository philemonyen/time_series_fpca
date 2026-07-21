import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from pathlib import Path
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.transformation.fda.fpca import fpca_with_param
from methods.transformation.fda.fica import compute_fica
from methods.transformation.nonlinear.isomap import find_optimal_k, find_optimal_manifold_dim
from methods.transformation.nonlinear.tsne import tsne_trasformation
from methods.transformation.nonlinear.diffusion_map import DenseDiffusionMap
from methods.transformation.nonlinear.umap import tune_umap
from methods.transformation.nonlinear.kpca import kpca_tune_n_components, kpca_with_param, tune_gamma
from methods.transformation.nonlinear.principal_curve import principal_curve
from methods.evaluation.fidelity import *
from methods.validation.dataset_creation import *

if __name__ == "__main__":
    ## ------------ Data Preparation ------------ ##
    diagnostic = "NORM"
    lead = 1
    n_data = 1000
    sr = get_sr()
    n_beats = 10
    domain_range = (0, 1)
    n_timepoints = n_beats * sr
    n_basis = int(n_timepoints / 2)
    n_components = 10
    landmark_locations = np.linspace(0, 1, n_beats+2)[1:-1]

    # Result save path
    save_path = f"images/fidelity_val/fpca/"
    path=Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Get Real Data
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    trimmed_real_fd, real_landmarks_all = extract_ecg_clinical_landmarks(real_all, n_beats, sr)

    real_fd = trimmed_real_fd[:n_data]
    real_landmarks = real_landmarks_all[:n_data]
    substitute_fd = trimmed_real_fd[n_data:2*n_data]
    substitute_landmarks = real_landmarks_all[n_data:2*n_data]

    # Create Controlled Flaw Dataset
    scenarios = ["oversmoothing", "memorization", "gaussian_noise", "mode_collapse_vary_modes", "mode_collapse_vary_spike_ratio", "segment_leaking"]
    datasets = {}
    # Result Tracking
    result_tracking = {}
    for scenario in scenarios:
        lmr = []
        scales = []
        if scenario == "oversmoothing":
            datasets = oversmoothing_creation(real_fd, real_landmarks)
        elif scenario == "memorization":
            datasets = memorization_creation(real_fd, substitute_fd, real_landmarks, substitute_landmarks)
        elif scenario == "gaussian_noise":
            datasets = gaussian_noise_creation(real_fd, real_landmarks)
        elif scenario == "mode_collapse_vary_modes":
            datasets = mode_collapse_vary_modes_creation(real_fd, real_landmarks)
        elif scenario == "mode_collapse_vary_spike_ratio":
            datasets = mode_collapse_vary_spike_ratio_creation(real_fd, real_landmarks)
        elif scenario == "segment_leaking":
            datasets = segment_leaking_creation(real_fd, substitute_fd, real_landmarks, substitute_landmarks)

        result_tracking[scenario] = {}

        for key, value in datasets.items():
            flaw_fd, flaw_landmarks = value
            #### ------------ Individual FPCA ------------ ####
            # Apply FPCA on Real dataset
            lambda_ = basis_smoothing_hyperparameter_tuning(trimmed_real_fd, n_basis, domain_range)
            real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(trimmed_real_fd, lambda_, n_basis, domain_range)
            real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks_all, landmark_locations)
            real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(real_aligned_fd, n_components)

            # Apply FPCA on flaw dataset
            lambda_ = basis_smoothing_hyperparameter_tuning(flaw_fd, n_basis, domain_range)
            flaw_fd_smooth, _, _, _ = basis_smoothing_with_lambda(flaw_fd, lambda_, n_basis, domain_range)
            flaw_aligned_fd, _ = landmark_registration(flaw_fd_smooth, flaw_landmarks, landmark_locations)
            # flaw_mean, flaw_components, flaw_scores, flaw_var_ratio, flaw_fpca_ = fpca_with_param(flaw_aligned_fd, n_components)

            ## Evaluation: Principal Component Alignment
            # pc_alignment_score = pc_alignment(real_components.data_matrix, flaw_components.data_matrix)

            #### ------------ Shared FPCA ------------ ####
            # Apply Real FPCA on Synthetic
            shared_flaw_scores = real_fpca_.transform(flaw_aligned_fd)

            # Apply FICA on real and synthetic FPC scores
            real_fica_scores, real_fica_components, real_ica = compute_fica(real_scores, real_components)
            synthetic_fica_scores, synthetic_fica_components, synthetic_ica = compute_fica(shared_flaw_scores, real_components)

            # Evaluation: MMD, Mahalanobis, FPC KS, PRDC, LMR
            fpca_score_mmd = mmd(real_scores, shared_flaw_scores)
            fpca_score_mahalanobis = mahalanobis(real_scores, shared_flaw_scores)
            # fpca_score_ks = kolmogorov_smirnov(real_scores, shared_flaw_scores)
            fpca_prdc = prdc(real_scores, shared_flaw_scores)
            fpca_lmr = local_mixing_ratio(real_scores, shared_flaw_scores)
            lmr.append(fpca_lmr)
            scales.append(key)
            
            # Apply Isomap on real and synthetic FPC scores separately
            # optimal_k_real = find_optimal_k(real_scores)
            # optimal_dim_real = find_optimal_manifold_dim(real_scores, optimal_k_real)
            # isomap_real = Isomap(n_neighbors=optimal_k_real, n_components=optimal_dim_real)
            # real_isomap_embedding = isomap_real.fit_transform(real_scores)

            # optimal_k_flaw = find_optimal_k(shared_flaw_scores)
            # optimal_dim_flaw = find_optimal_manifold_dim(shared_flaw_scores, optimal_k_flaw)
            # isomap_flaw = Isomap(n_neighbors=optimal_k_flaw, n_components=optimal_dim_flaw)
            # flaw_isomap_embedding = isomap_flaw.fit_transform(shared_flaw_scores)

            # ## Evaluation: Gromov Wasserstein & Procrustes Analysis
            # isomap_gw = gromov_wasserstein(real_isomap_embedding, flaw_isomap_embedding)
            # isomap_procrustes = unpaired_procrustes(real_isomap_embedding, flaw_isomap_embedding)

            # Apply t-SNE on real and synthetic FPC scores separately
            # real_tsne_embedding = tsne_trasformation(real_scores)
            # flaw_tsne_embedding = tsne_trasformation(shared_flaw_scores)

            ## Evaluation: Gromov Wasserstein
            # tsne_gw = gromov_wasserstein(real_tsne_embedding, flaw_tsne_embedding)

            # Apply Diffusion Map on real and synthetic FPC scores separately
            real_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(real_scores)
            real_dmap_evals = real_dmap.evals_
            real_dmap_embedding = real_dmap.transform(real_scores)

            # flaw_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(shared_flaw_scores)
            # flaw_dmap_evals = flaw_dmap.evals_
            # flaw_dmap_embedding = flaw_dmap.transform(shared_flaw_scores)

            # ## Evaluation: Gromov Wasserstein & RMSE on Von Neumann Entropy Curve
            # dmap_gw = gromov_wasserstein(real_dmap_embedding, flaw_dmap_embedding)
            # dmap_entropy_rmse = diffusion_map_entropy_rmse(real_dmap_evals, flaw_dmap_evals)

            # Apply Diffusion Map on real and transform synthetic 
            flaw_dmap_embedding_shared_real = real_dmap.transform(shared_flaw_scores)

            ## Evaluation: JS Divergence, MMD, PRDC, LMR
            dmap_js_divergence = grid_js_divergence(real_dmap_embedding, flaw_dmap_embedding_shared_real)
            dmap_mmd = mmd(real_dmap_embedding, flaw_dmap_embedding_shared_real)
            # dmap_prdc = prdc(real_dmap_embedding, flaw_dmap_embedding_shared_real)
            # dmap_lmr = local_mixing_ratio(real_dmap_embedding, flaw_dmap_embedding_shared_real)

            # Apply UMAP on real and synthetic FPC scores separately
            real_umap = tune_umap(real_scores)
            real_umap_embedding = real_umap.transform(real_scores)
            # flaw_umap = tune_umap(shared_flaw_scores)
            # flaw_umap_embedding = flaw_umap.transform(shared_flaw_scores)

            ## Evaluation: Gromov Wasserstein
            # umap_gw = gromov_wasserstein(real_umap_embedding, flaw_umap_embedding)

            # Apply UMAP on real and transform synthetic 
            flaw_umap_embedding_shared_real = real_umap.transform(shared_flaw_scores)

            ## Evaluation: JS Divergence, MMD, PRDC, LMR
            umap_js_divergence = grid_js_divergence(real_umap_embedding, flaw_umap_embedding_shared_real)
            umap_mmd = mmd(real_umap_embedding, flaw_umap_embedding_shared_real)
            # umap_prdc = prdc(real_umap_embedding, flaw_umap_embedding_shared_real)
            # umap_lmr = local_mixing_ratio(real_umap_embedding, flaw_umap_embedding_shared_real)

            # # Apply kPCA on real and transform synthetic 
            # real_kpca_n_components = kpca_tune_n_components(real_scores)
            # real_kpca_gamma = tune_gamma(real_scores)
            # real_kpca_embedding, real_kpca = kpca_with_param(real_scores, real_kpca_n_components, real_kpca_gamma)
            # flaw_kpca_embedding = real_kpca.transform(shared_flaw_scores)

            # ## Evaluation: MMD, Mahalanobis, FPC KS, PRDC, LMR
            # kpca_score_mmd = mmd(real_kpca_embedding, flaw_kpca_embedding)
            # kpca_score_mahalanobis = mahalanobis(real_kpca_embedding, flaw_kpca_embedding)
            # kpca_score_ks = kolmogorov_smirnov(real_kpca_embedding, flaw_kpca_embedding)
            # kpca_prdc = prdc(real_kpca_embedding, flaw_kpca_embedding)
            # kpca_lmr = local_mixing_ratio(real_kpca_embedding, flaw_kpca_embedding)

            # # Get Principal Curves of real and synthetic
            # real_principal_curve = principal_curve(real_scores, real_components[0], real_mean)
            # flaw_principal_curve = principal_curve(shared_flaw_scores, flaw_components[0], flaw_mean)

            ## Evaluation: Wasserstein Distance
            # principal_curve_wd = wasserstein(real_principal_curve.data_matrix.squeeze(), flaw_principal_curve.data_matrix.squeeze())

            #### ------------ Result Display ------------ ####
            result_tracking[scenario][key] = {}
            ## Individual FPCA
            # # Principal Component Alignment
            # result_tracking[scenario][key]['pc_alignment_score'] = pc_alignment_score['component_wise_cosine_sim']
            # result_tracking[scenario][key]['mean_cosine_similarity'] = pc_alignment_score['mean_cosine_similarity']
            # result_tracking[scenario][key]['subspace_overlap_score'] = pc_alignment_score['subspace_overlap_score']

            ## Shared FPCA
            # FPC Score MMD, Mahalanobis, FPC KS, PRDC, LMR
            result_tracking[scenario][key]['fpca_score_mmd'] = fpca_score_mmd
            result_tracking[scenario][key]['fpca_score_mahalanobis'] = fpca_score_mahalanobis
            # result_tracking[scenario][key]['fpca_score_ks'] = fpca_score_ks
            # print("Shared FPCA: MMD")
            # print(f"    MMD: {fpca_score_mmd}")
            # print(f"    Mahalanobis: {fpca_score_mahalanobis}")
            # print(f"    FPC KS: {fpca_score_ks}")

            plt.plot(fpca_prdc[4], fpca_prdc[0], label="Precision")
            plt.plot(fpca_prdc[4], fpca_prdc[1], label="Recall")
            plt.plot(fpca_prdc[4], fpca_prdc[2], label="Density")
            plt.plot(fpca_prdc[4], fpca_prdc[3], label="Coverage")
            plt.xlabel("Number of Neighbors")
            plt.ylabel("Score")
            plt.title(f"FPCA PRDC: {scenario}, Flaw Scale: {key}")
            plt.legend()
            plt.savefig(save_path + f"FPCA_PRDC_{scenario}_{key}.png")
            plt.close()
            
            # plt.plot(fpca_lmr[2], fpca_lmr[0], label="Ratio")
            # plt.axhline(y=fpca_lmr[1], color='red', linestyle='--', label="Baseline")
            # plt.xlabel("Number of Neighbors")
            # plt.ylabel("Local Mixing Ratio")
            # plt.title(f"FPCA LMR: {scenario}, Flaw Scale: {key}")
            # plt.legend()
            # plt.savefig(save_path + f"FPCA_LMR_{scenario}_{key}.png")
            # plt.close()

            # Isomap: Gromov Wasserstein & Procrustes Analysis
            # result_tracking[scenario][key]['isomap_gw'] = isomap_gw
            # result_tracking[scenario][key]['isomap_procrustes'] = isomap_procrustes['unpaired_similarity_score']

            # t-SNE: Gromov Wasserstein
            # result_tracking[scenario][key]['tsne_gw'] = tsne_gw

            # Individual Diffusion Map: Gromov Wasserstein & RMSE on Von Neumann Entropy Curve
            # result_tracking[scenario][key]['dmap_gw'] = dmap_gw
            # result_tracking[scenario][key]['dmap_entropy_rmse'] = dmap_entropy_rmse['entropy_rmse']

            # Shared Diffusion Map: JS Divergence, MMD, PRDC, LMR
            result_tracking[scenario][key]['dmap_js_divergence'] = dmap_js_divergence
            result_tracking[scenario][key]['dmap_mmd'] = dmap_mmd

            # plt.plot(dmap_prdc[4], dmap_prdc[0], label="Precision")
            # plt.plot(dmap_prdc[4], dmap_prdc[1], label="Recall")
            # plt.plot(dmap_prdc[4], dmap_prdc[2], label="Density")
            # plt.plot(dmap_prdc[4], dmap_prdc[3], label="Coverage")
            # plt.xlabel("Number of Neighbors")
            # plt.ylabel("Score")
            # plt.title(f"DMap PRDC: {scenario}, Flaw Scale: {key}")
            # plt.legend()
            # plt.savefig(save_path + f"DMap_PRDC_{scenario}_{key}.png")
            # plt.close()
# 
            # plt.plot(dmap_lmr[2], dmap_lmr[0], label="Ratio")
            # plt.axhline(y=dmap_lmr[1], color='red', linestyle='--', label="Baseline")
            # plt.xlabel("Number of Neighbors")
            # plt.ylabel("Local Mixing Ratio")
            # plt.title(f"DMap LMR: {scenario}, Flaw Scale: {key}")
            # plt.legend()
            # plt.savefig(save_path + f"DMap_LMR_{scenario}_{key}.png")
            # plt.close()

            # Individual UMAP: Gromov Wasserstein
            # result_tracking[scenario][key]['umap_gw'] = umap_gw

            # Shared UMAP: JS Divergence, MMD, PRDC, LMR
            result_tracking[scenario][key]['umap_js_divergence'] = umap_js_divergence
            result_tracking[scenario][key]['umap_mmd'] = umap_mmd

            plt.scatter(real_umap_embedding[:, 0], real_umap_embedding[:, 1], label="Real")
            plt.scatter(flaw_umap_embedding_shared_real[:, 0], flaw_umap_embedding_shared_real[:, 1], label="Flaw")
            plt.title(f"UMAP Embedding: {scenario}, Flaw Scale: {key}")
            plt.legend()
            plt.savefig(save_path + f"UMAP_Embedding_{scenario}_{key}.png")
            plt.close()

            # plt.plot(umap_prdc[4], umap_prdc[0], label="Precision")
            # plt.plot(umap_prdc[4], umap_prdc[1], label="Recall")
            # plt.plot(umap_prdc[4], umap_prdc[2], label="Density")
            # plt.plot(umap_prdc[4], umap_prdc[3], label="Coverage")
            # plt.xlabel("Number of Neighbors")
            # plt.ylabel("Score")
            # plt.title(f"UMAP PRDC: {scenario}, Flaw Scale: {key}")
            # plt.legend()
            # plt.savefig(save_path + f"UMAP_PRDC_{scenario}_{key}.png")
            # plt.close()

            # plt.plot(umap_lmr[2], umap_lmr[0], label="Ratio")
            # plt.axhline(y=umap_lmr[1], color='red', linestyle='--', label="Baseline")
            # plt.xlabel("Number of Neighbors")
            # plt.ylabel("Local Mixing Ratio")
            # plt.title(f"UMAP LMR: {scenario}, Flaw Scale: {key}")
            # plt.legend()
            # plt.savefig(save_path + f"UMAP_LMR_{scenario}_{key}.png")
            # plt.close()

            # kPCA: MMD, Mahalanobis, FPC KS, PRDC, LMR
            # result_tracking[scenario][key]['kpca_score_mmd'] = kpca_score_mmd
            # result_tracking[scenario][key]['kpca_score_mahalanobis'] = kpca_score_mahalanobis
            # result_tracking[scenario][key]['kpca_score_ks'] = kpca_score_ks

            # plt.plot(kpca_prdc[4], kpca_prdc[0], label="Precision")
            # plt.plot(kpca_prdc[4], kpca_prdc[1], label="Recall")
            # plt.plot(kpca_prdc[4], kpca_prdc[2], label="Density")
            # plt.plot(kpca_prdc[4], kpca_prdc[3], label="Coverage")
            # plt.xlabel("Number of Neighbors")
            # plt.ylabel("Score")
            # plt.title(f"kPCA PRDC: {scenario}, Flaw Scale: {key}")
            # plt.legend()
            # plt.savefig(save_path + f"kPCA_PRDC_{scenario}_{key}.png")
            # plt.close()

            # plt.plot(kpca_lmr[2], kpca_lmr[0], label="Ratio")
            # plt.axhline(y=kpca_lmr[1], color='red', linestyle='--', label="Baseline")
            # plt.xlabel("Number of Neighbors")
            # plt.ylabel("Local Mixing Ratio")
            # plt.title(f"kPCA LMR: {scenario}, Flaw Scale: {key}")
            # plt.legend()
            # plt.savefig(save_path + f"kPCA_LMR_{scenario}_{key}.png")
            # plt.close()

            # Principal Curve: Wasserstein Distance
            # result_tracking[scenario][key]['principal_curve_wd'] = principal_curve_wd

            ## FICA IC plotting
            for i in range(n_components):
                plt.plot(real_fica_components[i].data_matrix.squeeze(), label="Real")
                plt.plot(synthetic_fica_components[i].data_matrix.squeeze(), label="Synthetic")
                plt.title(f"FICA Component {i}: {scenario}, Flaw Scale: {key}")
                plt.xlabel("Time")
                plt.ylabel("Variance")
                plt.legend()
                plt.savefig(save_path + f"FICA_Component_{scenario}_{key}_{i}.png")
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
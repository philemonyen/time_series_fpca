import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks, load_synthetic_dataset
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.transformation.fda.fica import compute_fica
from methods.transformation.fda.fpca import fpca_with_param
from methods.transformation.nonlinear.diffusion_map import DenseDiffusionMap
from methods.transformation.nonlinear.umap import tune_umap
from methods.evaluation.fidelity import *

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
    save_path = f"images/fidelity_eval/fpca/"
    path=Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # Get Real Data
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    trimmed_real_fd, real_landmarks_all = extract_ecg_clinical_landmarks(real_all, n_beats, sr)

    # Get Synthetic Data
    synthetic_all = load_synthetic_dataset(diagnostic, lead)
    trimmed_synthetic_fd, synthetic_landmarks_all = extract_ecg_clinical_landmarks(synthetic_all, n_beats, sr)

    #### ------------ Individual FPCA ------------ ####
    # Apply FPCA on Real dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(trimmed_real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(trimmed_real_fd, lambda_, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks_all, landmark_locations)
    real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(real_aligned_fd, n_components)

    # Apply FPCA on synthetic dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(trimmed_synthetic_fd, n_basis, domain_range)
    synthetic_fd_smooth, _, _, _ = basis_smoothing_with_lambda(trimmed_synthetic_fd, lambda_, n_basis, domain_range)
    synthetic_aligned_fd, _ = landmark_registration(synthetic_fd_smooth, synthetic_landmarks_all, landmark_locations)

    #### ------------ Shared FPCA ------------ ####
    # Apply Real FPCA on Synthetic
    synthetic_scores_shared_fpca = real_fpca_.transform(synthetic_aligned_fd)

    # Evaluation: MMD, Mahalanobis, FPC KS, PRDC, LMR
    fpca_score_mmd = mmd(real_scores, synthetic_scores_shared_fpca)
    fpca_score_mahalanobis = mahalanobis(real_scores, synthetic_scores_shared_fpca)
    fpca_prdc = prdc(real_scores, synthetic_scores_shared_fpca)
    fpca_lmr = local_mixing_ratio(real_scores, synthetic_scores_shared_fpca)

    # Apply Diffusion Map on real and synthetic FPC scores separately
    real_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(real_scores)
    real_dmap_evals = real_dmap.evals_
    real_dmap_embedding = real_dmap.transform(real_scores)

    # Apply Diffusion Map on real and transform synthetic 
    synthetic_dmap_embedding_shared_real = real_dmap.transform(synthetic_scores_shared_fpca)

    ## Evaluation: JS Divergence, MMD, PRDC, LMR
    dmap_js_divergence = grid_js_divergence(real_dmap_embedding, synthetic_dmap_embedding_shared_real)
    dmap_mmd = mmd(real_dmap_embedding, synthetic_dmap_embedding_shared_real)

    # Apply UMAP on real and synthetic FPC scores separately
    real_umap = tune_umap(real_scores)
    real_umap_embedding = real_umap.transform(real_scores)

    # Apply UMAP on real and transform synthetic 
    synthetic_umap_embedding_shared_real = real_umap.transform(synthetic_scores_shared_fpca)

    ## Evaluation: JS Divergence, MMD, PRDC, LMR
    umap_js_divergence = grid_js_divergence(real_umap_embedding, synthetic_umap_embedding_shared_real)
    umap_mmd = mmd(real_umap_embedding, synthetic_umap_embedding_shared_real)

    ## Individual FICA to visualize IC differences
    # Apply FICA on Real dataset
    # real_fica_scores, real_fica_components, real_ica = compute_fica(real_scores, real_components)
    # # Apply FICA on synthetic dataset
    # synthetic_fica_scores, synthetic_fica_components, synthetic_ica = compute_fica(synthetic_scores_shared_fpca, real_components)

    # Evaluation on raw aligned data
    mmd_score = mmd(real_aligned_fd.data_matrix.squeeze(), synthetic_aligned_fd.data_matrix.squeeze())
    wasserstein_score = wasserstein(real_aligned_fd.data_matrix.squeeze(), synthetic_aligned_fd.data_matrix.squeeze())

    # Apply UMAP on raw aligned data
    real_umap = tune_umap(real_aligned_fd.data_matrix.squeeze())
    raw_real_umap_embedding = real_umap.transform(real_aligned_fd.data_matrix.squeeze())
    raw_synthetic_umap_embedding = real_umap.transform(synthetic_aligned_fd.data_matrix.squeeze())

    #### ------------ Result Display ------------ ####
    f = open(save_path + "fpca_result.txt", "w")

    # Raw Aligned Data: MMD, Wasserstein
    f.write("Raw Aligned Data: MMD, Wasserstein\n")
    f.write(f"    MMD: {mmd_score}\n")
    f.write(f"    Wasserstein: {wasserstein_score}\n")
    
    ## Shared FPCA
    # FPC Score MMD, Mahalanobis, FPC KS, PRDC, LMR
    f.write("Shared FPCA: MMD\n")
    f.write(f"    MMD: {fpca_score_mmd}\n")
    f.write(f"    Mahalanobis: {fpca_score_mahalanobis}\n")

    plt.plot(fpca_prdc[4], fpca_prdc[0], label="Precision")
    plt.plot(fpca_prdc[4], fpca_prdc[1], label="Recall")
    plt.plot(fpca_prdc[4], fpca_prdc[2], label="Density")
    plt.plot(fpca_prdc[4], fpca_prdc[3], label="Coverage")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Score")
    plt.title("FPCA PRDC")
    plt.legend()
    plt.savefig(save_path + "FPCA_PRDC.png")
    plt.close()

    plt.plot(fpca_lmr[2], fpca_lmr[0], label="Ratio")
    plt.axhline(y=fpca_lmr[1], color='red', linestyle='--', label="Baseline")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Local Mixing Ratio")
    plt.title("FPCA Local Mixing Ratio")
    plt.legend()
    plt.savefig(save_path + "FPCA_LMR.png")
    plt.close()

    # Shared Diffusion Map: JS Divergence, MMD, PRDC, LMR
    f.write("Diffusion Map: JS Divergence, MMD, PRDC, LMR\n")
    f.write(f"    JS Divergence: {dmap_js_divergence}\n")
    f.write(f"    MMD: {dmap_mmd}\n")

    # Shared UMAP: JS Divergence, MMD, PRDC, LMR
    f.write("UMAP: JS Divergence, MMD, PRDC, LMR\n")
    f.write(f"    JS Divergence: {umap_js_divergence}\n")
    f.write(f"    MMD: {umap_mmd}\n")
    f.close()

    plt.scatter(real_umap_embedding[:, 0], real_umap_embedding[:, 1], label="Real")
    plt.scatter(synthetic_umap_embedding_shared_real[:, 0], synthetic_umap_embedding_shared_real[:, 1], label="Synthetic")
    plt.title("UMAP Embedding")
    plt.legend()
    plt.savefig(save_path + "UMAP_Embedding.png")
    plt.close()

    ## Raw UMAP plotting
    plt.scatter(raw_real_umap_embedding[:, 0], raw_real_umap_embedding[:, 1], label="Real")
    plt.scatter(raw_synthetic_umap_embedding[:, 0], raw_synthetic_umap_embedding[:, 1], label="Synthetic")
    plt.title("Raw UMAP Embedding")
    plt.legend()
    plt.savefig(save_path + "Raw_UMAP_Embedding.png")
    plt.close()

    ## FICA IC plotting
    # for i in range(n_components):
    #     plt.plot(real_fica_components[i].data_matrix.squeeze(), label="Real")
    #     plt.xlabel("Time")
    #     plt.ylabel("Variance")
    #     plt.title(f"FICA Real Component {i}")
    #     plt.legend()
    #     plt.savefig(save_path + f"FICA_Real_Component_{i}.png")
    #     plt.close()
    #     plt.plot(synthetic_fica_components[i].data_matrix.squeeze(), label="Synthetic")
    #     plt.xlabel("Time")
    #     plt.ylabel("Variance")
    #     plt.title(f"FICA Synthetic Component {i}")
    #     plt.legend()
    #     plt.savefig(save_path + f"FICA_Synthetic_Component_{i}.png")
    #     plt.close()
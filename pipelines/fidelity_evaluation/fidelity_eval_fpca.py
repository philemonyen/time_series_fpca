import readline
import numpy as np
import tabulate as tb
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from pathlib import Path
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks, load_synthetic_dataset
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.transformation.fda.fpca import fpca_with_param
from methods.transformation.nonlinear.isomap import find_optimal_k, find_optimal_manifold_dim
from methods.transformation.nonlinear.tsne import tsne_trasformation
from methods.transformation.nonlinear.diffusion_map import DenseDiffusionMap
from methods.transformation.nonlinear.umap import tune_umap
from methods.transformation.nonlinear.kpca import kpca_tune_n_components, kpca_with_param, tune_gamma
from methods.transformation.nonlinear.principal_curve import principal_curve
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
    synthetic_fd = trimmed_synthetic_fd[:n_data]
    synthetic_landmarks = synthetic_landmarks_all[:n_data]

    #### ------------ Individual FPCA ------------ ####
    # Apply FPCA on Real dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(trimmed_real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(trimmed_real_fd, lambda_, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks_all, landmark_locations)
    real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(real_aligned_fd, n_components)

    # Apply FPCA on synthetic dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(synthetic_fd, n_basis, domain_range)
    synthetic_fd_smooth, _, _, _ = basis_smoothing_with_lambda(synthetic_fd, lambda_, n_basis, domain_range)
    synthetic_aligned_fd, _ = landmark_registration(synthetic_fd_smooth, synthetic_landmarks_all, landmark_locations)
    synthetic_mean, synthetic_components, synthetic_scores, synthetic_var_ratio, synthetic_fpca_ = fpca_with_param(synthetic_aligned_fd, n_components)

    ## Evaluation: Principal Component Alignment
    pc_alignment_score = pc_alignment(real_components.data_matrix, synthetic_components.data_matrix)

    #### ------------ Shared FPCA ------------ ####
    # Apply Real FPCA on Synthetic
    synthetic_scores_shared_fpca = real_fpca_.transform(synthetic_aligned_fd)

    # Evaluation: MMD, Mahalanobis, FPC KS, PRDC, LMR
    fpca_score_mmd = mmd(real_scores, synthetic_scores_shared_fpca)
    fpca_score_mahalanobis = mahalanobis(real_scores, synthetic_scores_shared_fpca)
    fpca_score_ks = kolmogorov_smirnov(real_scores, synthetic_scores_shared_fpca)
    fpca_prdc = prdc(real_scores, synthetic_scores_shared_fpca)
    fpca_lmr = local_mixing_ratio(real_scores, synthetic_scores_shared_fpca)
    
    # Apply Isomap on real and synthetic FPC scores separately
    optimal_k_real = find_optimal_k(real_scores)
    optimal_dim_real = find_optimal_manifold_dim(real_scores, optimal_k_real)
    isomap_real = Isomap(n_neighbors=optimal_k_real, n_components=optimal_dim_real)
    real_isomap_embedding = isomap_real.fit_transform(real_scores)

    optimal_k_synthetic = find_optimal_k(synthetic_scores_shared_fpca)
    optimal_dim_synthetic = find_optimal_manifold_dim(synthetic_scores_shared_fpca, optimal_k_synthetic)
    isomap_synthetic = Isomap(n_neighbors=optimal_k_synthetic, n_components=optimal_dim_synthetic)
    synthetic_isomap_embedding = isomap_synthetic.fit_transform(synthetic_scores_shared_fpca)

    ## Evaluation: Gromov Wasserstein & Procrustes Analysis
    isomap_gw = gromov_wasserstein(real_isomap_embedding, synthetic_isomap_embedding)
    isomap_procrustes = unpaired_procrustes(real_isomap_embedding, synthetic_isomap_embedding)

    # Apply t-SNE on real and synthetic FPC scores separately
    real_tsne_embedding = tsne_trasformation(real_scores)
    synthetic_tsne_embedding = tsne_trasformation(synthetic_scores_shared_fpca)

    ## Evaluation: Gromov Wasserstein
    tsne_gw = gromov_wasserstein(real_tsne_embedding, synthetic_tsne_embedding)

    # Apply Diffusion Map on real and synthetic FPC scores separately
    real_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(real_scores)
    real_dmap_evals = real_dmap.evals_
    real_dmap_embedding = real_dmap.transform(real_scores)
    synthetic_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(synthetic_scores_shared_fpca)
    synthetic_dmap_evals = synthetic_dmap.evals_
    synthetic_dmap_embedding = synthetic_dmap.transform(synthetic_scores_shared_fpca)

    ## Evaluation: Gromov Wasserstein & RMSE on Von Neumann Entropy Curve
    dmap_gw = gromov_wasserstein(real_dmap_embedding, synthetic_dmap_embedding)
    dmap_entropy_rmse = diffusion_map_entropy_rmse(real_dmap_evals, synthetic_dmap_evals)

    # Apply Diffusion Map on real and transform synthetic 
    synthetic_dmap_embedding_shared_real = real_dmap.transform(synthetic_scores_shared_fpca)

    ## Evaluation: JS Divergence, MMD, PRDC, LMR
    dmap_js_divergence = grid_js_divergence(real_dmap_embedding, synthetic_dmap_embedding_shared_real)
    dmap_mmd = mmd(real_dmap_embedding, synthetic_dmap_embedding_shared_real)
    dmap_prdc = prdc(real_dmap_embedding, synthetic_dmap_embedding_shared_real)
    dmap_lmr = local_mixing_ratio(real_dmap_embedding, synthetic_dmap_embedding_shared_real)

    # Apply UMAP on real and synthetic FPC scores separately
    real_umap = tune_umap(real_scores)
    real_umap_embedding = real_umap.transform(real_scores)
    synthetic_umap = tune_umap(synthetic_scores_shared_fpca)
    synthetic_umap_embedding = synthetic_umap.transform(synthetic_scores)

    ## Evaluation: Gromov Wasserstein
    umap_gw = gromov_wasserstein(real_umap_embedding, synthetic_umap_embedding)

    # Apply UMAP on real and transform synthetic 
    synthetic_umap_embedding_shared_real = real_umap.transform(synthetic_scores_shared_fpca)

    ## Evaluation: JS Divergence, MMD, PRDC, LMR
    umap_js_divergence = grid_js_divergence(real_umap_embedding, synthetic_umap_embedding_shared_real)
    umap_mmd = mmd(real_umap_embedding, synthetic_umap_embedding_shared_real)
    umap_prdc = prdc(real_umap_embedding, synthetic_umap_embedding_shared_real)
    umap_lmr = local_mixing_ratio(real_umap_embedding, synthetic_umap_embedding_shared_real)

    # Apply kPCA on real and transform synthetic 
    real_kpca_n_components = kpca_tune_n_components(real_scores)
    real_kpca_gamma = tune_gamma(real_scores)
    real_kpca_embedding, real_kpca = kpca_with_param(real_scores, real_kpca_n_components, real_kpca_gamma)
    synthetic_kpca_embedding = real_kpca.transform(synthetic_scores_shared_fpca)

    ## Evaluation: MMD, Mahalanobis, FPC KS, PRDC, LMR
    kpca_score_mmd = mmd(real_kpca_embedding, synthetic_kpca_embedding)
    kpca_score_mahalanobis = mahalanobis(real_kpca_embedding, synthetic_kpca_embedding)
    kpca_score_ks = kolmogorov_smirnov(real_kpca_embedding, synthetic_kpca_embedding)
    kpca_prdc = prdc(real_kpca_embedding, synthetic_kpca_embedding)
    kpca_lmr = local_mixing_ratio(real_kpca_embedding, synthetic_kpca_embedding)

    # Get Principal Curves of real and synthetic
    real_principal_curve = principal_curve(real_scores, real_components[0], real_mean)
    synthetic_principal_curve = principal_curve(synthetic_scores_shared_fpca, synthetic_components[0], synthetic_mean)

    ## Evaluation: Wasserstein Distance
    principal_curve_wd = wasserstein(real_principal_curve.data_matrix.squeeze(), synthetic_principal_curve.data_matrix.squeeze())

    #### ------------ Result Display ------------ ####
    ## Individual FPCA
    for i in range(n_components):
        plt.plot(real_components[i].data_matrix.squeeze(), label="Real")
        plt.plot(synthetic_components[i].data_matrix.squeeze(), label="Synthetic")
        plt.legend()
        plt.title(f"FPCA Component {i}")
        plt.xlabel("Time")
        plt.ylabel("Variance")
        plt.savefig(save_path + f"Individual_FPCA_Component_{i}.png")
        plt.close()

    # Principal Component Alignment
    f = open(save_path + "fpca_result.txt", "w")
    f.write("Individual FPCA: Principal Component Alignment\n")
    f.write(f"    Component-wise Cosine Similarity: {pc_alignment_score['component_wise_cosine_sim']}\n")
    f.write(f"    Mean Cosine Similarity: {pc_alignment_score['mean_cosine_similarity']}\n")
    f.write(f"    Subspace Overlap Score: {pc_alignment_score['subspace_overlap_score']}\n")

    ## Shared FPCA
    # FPC Score MMD, Mahalanobis, FPC KS, PRDC, LMR
    f.write("Shared FPCA: MMD\n")
    f.write(f"    MMD: {fpca_score_mmd}\n")
    f.write(f"    Mahalanobis: {fpca_score_mahalanobis}\n")
    f.write(f"    FPC KS: {fpca_score_ks}\n")

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

    # Isomap: Gromov Wasserstein & Procrustes Analysis
    f.write("Isomap: Gromov Wasserstein & Procrustes Analysis\n")
    f.write(f"    Gromov Wasserstein: {isomap_gw}\n")
    f.write(f"    Procrustes Similarity: {isomap_procrustes['unpaired_similarity_score']}\n")

    # t-SNE: Gromov Wasserstein
    f.write("t-SNE: Gromov Wasserstein\n")
    f.write(f"    Gromov Wasserstein: {tsne_gw}\n")

    # Individual Diffusion Map: Gromov Wasserstein & RMSE on Von Neumann Entropy Curve
    f.write("Diffusion Map: Gromov Wasserstein & RMSE on Von Neumann Entropy Curve\n")
    f.write(f"    Gromov Wasserstein: {dmap_gw}\n")
    f.write(f"    RMSE on Von Neumann Entropy Curve: {dmap_entropy_rmse['entropy_rmse']}\n")

    # Shared Diffusion Map: JS Divergence, MMD, PRDC, LMR
    f.write("Diffusion Map: JS Divergence, MMD, PRDC, LMR\n")
    f.write(f"    JS Divergence: {dmap_js_divergence}\n")
    f.write(f"    MMD: {dmap_mmd}\n")

    plt.plot(dmap_prdc[4], dmap_prdc[0], label="Precision")
    plt.plot(dmap_prdc[4], dmap_prdc[1], label="Recall")
    plt.plot(dmap_prdc[4], dmap_prdc[2], label="Density")
    plt.plot(dmap_prdc[4], dmap_prdc[3], label="Coverage")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Score")
    plt.title("Diffusion Map PRDC")
    plt.legend()
    plt.savefig(save_path + "DMap_PRDC.png")
    plt.close()

    plt.plot(dmap_lmr[2], dmap_lmr[0], label="Ratio")
    plt.axhline(y=dmap_lmr[1], color='red', linestyle='--', label="Baseline")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Local Mixing Ratio")
    plt.title("Diffusion Map Local Mixing Ratio")
    plt.legend()
    plt.savefig(save_path + "DMap_LMR.png")
    plt.close()

    # Individual UMAP: Gromov Wasserstein
    f.write("UMAP: Gromov Wasserstein\n")
    f.write(f"    Gromov Wasserstein: {umap_gw}\n")

    # Shared UMAP: JS Divergence, MMD, PRDC, LMR
    f.write("UMAP: JS Divergence, MMD, PRDC, LMR\n")
    f.write(f"    JS Divergence: {umap_js_divergence}\n")
    f.write(f"    MMD: {umap_mmd}\n")

    plt.plot(umap_prdc[4], umap_prdc[0], label="Precision")
    plt.plot(umap_prdc[4], umap_prdc[1], label="Recall")
    plt.plot(umap_prdc[4], umap_prdc[2], label="Density")
    plt.plot(umap_prdc[4], umap_prdc[3], label="Coverage")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Score")
    plt.title("UMAP PRDC")
    plt.legend()
    plt.savefig(save_path + "UMAP_PRDC.png")
    plt.close()

    plt.plot(umap_lmr[2], umap_lmr[0], label="Ratio")
    plt.axhline(y=umap_lmr[1], color='red', linestyle='--', label="Baseline")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Local Mixing Ratio")
    plt.title("UMAP Local Mixing Ratio")
    plt.legend()
    plt.savefig(save_path + "UMAP_LMR.png")
    plt.close()

    # kPCA: MMD, Mahalanobis, FPC KS, PRDC, LMR
    f.write("kPCA: MMD, Mahalanobis, FPC KS, PRDC, LMR\n")
    f.write(f"    MMD: {kpca_score_mmd}\n")
    f.write(f"    Mahalanobis: {kpca_score_mahalanobis}\n")
    f.write(f"    FPC KS: {kpca_score_ks}\n")

    plt.plot(kpca_prdc[4], kpca_prdc[0], label="Precision")
    plt.plot(kpca_prdc[4], kpca_prdc[1], label="Recall")
    plt.plot(kpca_prdc[4], kpca_prdc[2], label="Density")
    plt.plot(kpca_prdc[4], kpca_prdc[3], label="Coverage")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Score")
    plt.title("kPCA PRDC")
    plt.legend()
    plt.savefig(save_path + "kPCA_PRDC.png")
    plt.close()

    plt.plot(kpca_lmr[2], kpca_lmr[0], label="Ratio")
    plt.axhline(y=kpca_lmr[1], color='red', linestyle='--', label="Baseline")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Local Mixing Ratio")
    plt.title("kPCA Local Mixing Ratio")
    plt.legend()
    plt.savefig(save_path + "kPCA_LMR.png")
    plt.close()

    # Principal Curve: Wasserstein Distance
    f.write("Principal Curve: Wasserstein Distance\n")
    f.write(f"    Wasserstein Distance: {principal_curve_wd}\n")
    f.close()
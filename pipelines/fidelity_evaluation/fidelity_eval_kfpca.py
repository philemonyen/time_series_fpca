import numpy as np
import tabulate as tb
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from pathlib import Path
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks, load_synthetic_dataset
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.transformation.fda.kfpca import kfpca_with_param, kfpca_tune_gamma, kfpca_tuning_n_components
from methods.transformation.nonlinear.isomap import find_optimal_k, find_optimal_manifold_dim
from methods.transformation.nonlinear.tsne import tsne_trasformation
from methods.transformation.nonlinear.diffusion_map import DenseDiffusionMap
from methods.transformation.nonlinear.umap import tune_umap
from methods.transformation.nonlinear.kpca import kpca_tune_n_components, kpca_with_param, tune_gamma
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
    save_path = f"images/fidelity_eval/kfpca/"
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

    #### ------------ Shared kFPCA ------------ ####
    # Apply kFPCA on Real dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(trimmed_real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(trimmed_real_fd, lambda_, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks_all, landmark_locations)
    kfpca_optimal_gamma = kfpca_tune_gamma(real_aligned_fd)
    kfpca_optimal_n_components = kfpca_tuning_n_components(real_aligned_fd)
    real_kfpca_embedding, real_kfpca = kfpca_with_param(real_aligned_fd, kfpca_optimal_n_components, kfpca_optimal_gamma)

    # Apply real kFPCA on synthetic dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(synthetic_fd, n_basis, domain_range)
    synthetic_fd_smooth, _, _, _ = basis_smoothing_with_lambda(synthetic_fd, lambda_, n_basis, domain_range)
    synthetic_aligned_fd, _ = landmark_registration(synthetic_fd_smooth, synthetic_landmarks_all, landmark_locations)
    synthetic_kfpca_embedding = real_kfpca.transform(synthetic_aligned_fd)

    ## Evaluation: MMD, Mahalanobis, FPC KS, PRDC, LMR
    kfpca_score_mmd = mmd(real_kfpca_embedding, synthetic_kfpca_embedding)
    kfpca_score_mahalanobis = mahalanobis(real_kfpca_embedding, synthetic_kfpca_embedding)
    kfpca_score_ks = kolmogorov_smirnov(real_kfpca_embedding, synthetic_kfpca_embedding)
    kfpca_prdc = prdc(real_kfpca_embedding, synthetic_kfpca_embedding)
    kfpca_lmr = local_mixing_ratio(real_kfpca_embedding, synthetic_kfpca_embedding)
    
    # Apply Isomap on real and synthetic kFPCA scores separately
    optimal_k_real = find_optimal_k(real_kfpca_embedding)
    optimal_dim_real = find_optimal_manifold_dim(real_kfpca_embedding, optimal_k_real)
    isomap_real = Isomap(n_neighbors=optimal_k_real, n_components=optimal_dim_real)
    real_isomap_embedding = isomap_real.fit_transform(real_kfpca_embedding)

    optimal_k_synthetic = find_optimal_k(synthetic_kfpca_embedding)
    optimal_dim_synthetic = find_optimal_manifold_dim(synthetic_kfpca_embedding, optimal_k_synthetic)
    isomap_synthetic = Isomap(n_neighbors=optimal_k_synthetic, n_components=optimal_dim_synthetic)
    synthetic_isomap_embedding = isomap_synthetic.fit_transform(synthetic_kfpca_embedding)

    ## Evaluation: Gromov Wasserstein & Procrustes Analysis
    isomap_gw = gromov_wasserstein(real_isomap_embedding, synthetic_isomap_embedding)
    isomap_procrustes = unpaired_procrustes(real_isomap_embedding, synthetic_isomap_embedding)

    # Apply t-SNE on real and synthetic kFPCA scores separately
    real_tsne_embedding = tsne_trasformation(real_kfpca_embedding)
    synthetic_tsne_embedding = tsne_trasformation(synthetic_kfpca_embedding)

    ## Evaluation: Gromov Wasserstein
    tsne_gw = gromov_wasserstein(real_tsne_embedding, synthetic_tsne_embedding)

    # Apply Diffusion Map on real and synthetic kFPCA scores separately
    real_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(real_kfpca_embedding)
    real_dmap_evals = real_dmap.evals_
    real_dmap_embedding = real_dmap.transform(real_kfpca_embedding)

    synthetic_dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(synthetic_kfpca_embedding)
    synthetic_dmap_evals = synthetic_dmap.evals_
    synthetic_dmap_embedding = synthetic_dmap.transform(synthetic_kfpca_embedding)

    ## Evaluation: Gromov Wasserstein & RMSE on Von Neumann Entropy Curve
    dmap_gw = gromov_wasserstein(real_dmap_embedding, synthetic_dmap_embedding)
    dmap_entropy_rmse = diffusion_map_entropy_rmse(real_dmap_evals, synthetic_dmap_evals)

    # Apply Diffusion Map on real and transform synthetic 
    synthetic_dmap_embedding_shared_real = real_dmap.transform(synthetic_kfpca_embedding)

    ## Evaluation: JS Divergence, MMD, PRDC, LMR
    dmap_js_divergence = grid_js_divergence(real_dmap_embedding, synthetic_dmap_embedding_shared_real)
    dmap_mmd = mmd(real_dmap_embedding, synthetic_dmap_embedding_shared_real)
    dmap_prdc = prdc(real_dmap_embedding, synthetic_dmap_embedding_shared_real)
    dmap_lmr = local_mixing_ratio(real_dmap_embedding, synthetic_dmap_embedding_shared_real)

    # Apply UMAP on real and synthetic kFPCA scores separately
    real_umap = tune_umap(real_kfpca_embedding)
    real_umap_embedding = real_umap.transform(real_kfpca_embedding)
    synthetic_umap = tune_umap(synthetic_kfpca_embedding)
    synthetic_umap_embedding = synthetic_umap.transform(synthetic_kfpca_embedding)

    ## Evaluation: Gromov Wasserstein
    umap_gw = gromov_wasserstein(real_umap_embedding, synthetic_umap_embedding)

    # Apply UMAP on real and transform synthetic 
    synthetic_umap_embedding_shared_real = real_umap.transform(synthetic_kfpca_embedding)

    ## Evaluation: JS Divergence, MMD, PRDC, LMR
    umap_js_divergence = grid_js_divergence(real_umap_embedding, synthetic_umap_embedding_shared_real)
    umap_mmd = mmd(real_umap_embedding, synthetic_umap_embedding_shared_real)
    umap_prdc = prdc(real_umap_embedding, synthetic_umap_embedding_shared_real)
    umap_lmr = local_mixing_ratio(real_umap_embedding, synthetic_umap_embedding_shared_real)

    # Apply kPCA on real and transform synthetic 
    real_kpca_n_components = kpca_tune_n_components(real_kfpca_embedding)
    real_kpca_gamma = tune_gamma(real_kfpca_embedding)
    real_kpca_embedding, real_kpca = kpca_with_param(real_kfpca_embedding, real_kpca_n_components, real_kpca_gamma)
    synthetic_kpca_embedding = real_kpca.transform(synthetic_kfpca_embedding)

    ## Evaluation: MMD, Mahalanobis, FPC KS, PRDC, LMR
    kpca_score_mmd = mmd(real_kpca_embedding, synthetic_kpca_embedding)
    kpca_score_mahalanobis = mahalanobis(real_kpca_embedding, synthetic_kpca_embedding)
    kpca_score_ks = kolmogorov_smirnov(real_kpca_embedding, synthetic_kpca_embedding)
    kpca_prdc = prdc(real_kpca_embedding, synthetic_kpca_embedding)
    kpca_lmr = local_mixing_ratio(real_kpca_embedding, synthetic_kpca_embedding)

    #### ------------ Result Display ------------ ####
    ## Shared kFPCA
    # FPC Score MMD, Mahalanobis, FPC KS, PRDC, LMR
    print("Shared kFPCA: MMD")
    print(f"    MMD: {kfpca_score_mmd}")
    print(f"    Mahalanobis: {kfpca_score_mahalanobis}")
    print(f"    FPC KS: {kfpca_score_ks}")

    plt.plot(kfpca_prdc[4], kfpca_prdc[0], label="Precision")
    plt.plot(kfpca_prdc[4], kfpca_prdc[1], label="Recall")
    plt.plot(kfpca_prdc[4], kfpca_prdc[2], label="Density")
    plt.plot(kfpca_prdc[4], kfpca_prdc[3], label="Coverage")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Score")
    plt.title("PRDC")
    plt.legend()
    plt.savefig(save_path + "kFPCA_PRDC.png")
    plt.close()

    plt.plot(kfpca_lmr[2], kfpca_lmr[0], label="Ratio")
    plt.axhline(y=kfpca_lmr[1], color='red', linestyle='--', label="Baseline")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Local Mixing Ratio")
    plt.title("Local Mixing Ratio")
    plt.legend()
    plt.savefig(save_path + "kFPCA_LMR.png")
    plt.close()

    # Isomap: Gromov Wasserstein & Procrustes Analysis
    print("Isomap: Gromov Wasserstein & Procrustes Analysis")
    print(f"    Gromov Wasserstein: {isomap_gw}")
    print(f"    Procrustes Similarity: {isomap_procrustes['unpaired_similarity_score']}")

    # t-SNE: Gromov Wasserstein
    print("t-SNE: Gromov Wasserstein")
    print(f"    Gromov Wasserstein: {tsne_gw}")

    # Individual Diffusion Map: Gromov Wasserstein & RMSE on Von Neumann Entropy Curve
    print("Diffusion Map: Gromov Wasserstein & RMSE on Von Neumann Entropy Curve")
    print(f"    Gromov Wasserstein: {dmap_gw}")
    print(f"    RMSE on Von Neumann Entropy Curve: {dmap_entropy_rmse['entropy_rmse']}")

    # Shared Diffusion Map: JS Divergence, MMD, PRDC, LMR
    print("Diffusion Map: JS Divergence, MMD, PRDC, LMR")
    print(f"    JS Divergence: {dmap_js_divergence}")
    print(f"    MMD: {dmap_mmd}")

    plt.plot(dmap_prdc[4], dmap_prdc[0], label="Precision")
    plt.plot(dmap_prdc[4], dmap_prdc[1], label="Recall")
    plt.plot(dmap_prdc[4], dmap_prdc[2], label="Density")
    plt.plot(dmap_prdc[4], dmap_prdc[3], label="Coverage")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Score")
    plt.title("PRDC")
    plt.legend()
    plt.savefig(save_path + "DMap_PRDC.png")
    plt.close()

    plt.plot(dmap_lmr[2], dmap_lmr[0], label="Ratio")
    plt.axhline(y=dmap_lmr[1], color='red', linestyle='--', label="Baseline")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Local Mixing Ratio")
    plt.title("Local Mixing Ratio")
    plt.legend()
    plt.savefig(save_path + "DMap_LMR.png")
    plt.close()

    # Individual UMAP: Gromov Wasserstein
    print("UMAP: Gromov Wasserstein")
    print(f"    Gromov Wasserstein: {umap_gw}")

    # Shared UMAP: JS Divergence, MMD, PRDC, LMR
    print("UMAP: JS Divergence, MMD, PRDC, LMR")
    print(f"    JS Divergence: {umap_js_divergence}")
    print(f"    MMD: {umap_mmd}")

    plt.plot(umap_prdc[4], umap_prdc[0], label="Precision")
    plt.plot(umap_prdc[4], umap_prdc[1], label="Recall")
    plt.plot(umap_prdc[4], umap_prdc[2], label="Density")
    plt.plot(umap_prdc[4], umap_prdc[3], label="Coverage")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Score")
    plt.title("PRDC")
    plt.legend()
    plt.savefig(save_path + "UMAP_PRDC.png")
    plt.close()

    plt.plot(umap_lmr[2], umap_lmr[0], label="Ratio")
    plt.axhline(y=umap_lmr[1], color='red', linestyle='--', label="Baseline")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Local Mixing Ratio")
    plt.title("Local Mixing Ratio")
    plt.legend()
    plt.savefig(save_path + "UMAP_LMR.png")
    plt.close()

    # kPCA: MMD, Mahalanobis, FPC KS, PRDC, LMR
    print("kPCA: MMD, Mahalanobis, FPC KS, PRDC, LMR")
    print(f"    MMD: {kpca_score_mmd}")
    print(f"    Mahalanobis: {kpca_score_mahalanobis}")
    print(f"    FPC KS: {kpca_score_ks}")

    plt.plot(kpca_prdc[4], kpca_prdc[0], label="Precision")
    plt.plot(kpca_prdc[4], kpca_prdc[1], label="Recall")
    plt.plot(kpca_prdc[4], kpca_prdc[2], label="Density")
    plt.plot(kpca_prdc[4], kpca_prdc[3], label="Coverage")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Score")
    plt.title("PRDC")
    plt.legend()
    plt.savefig(save_path + "kPCA_PRDC.png")
    plt.close()

    plt.plot(kpca_lmr[2], kpca_lmr[0], label="Ratio")
    plt.axhline(y=kpca_lmr[1], color='red', linestyle='--', label="Baseline")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Local Mixing Ratio")
    plt.title("Local Mixing Ratio")
    plt.legend()
    plt.savefig(save_path + "kPCA_LMR.png")
    plt.close()
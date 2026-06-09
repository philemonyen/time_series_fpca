import numpy as np
import tabulate as tb
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from pathlib import Path
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks, load_synthetic_dataset
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.fpca import fpca_with_param
from methods.isomap import find_optimal_k, find_optimal_manifold_dim
from methods.fidelity_evaluation import mmd_distance, frechet_wasserstein, covariance_operator_dist, compute_prdc, kolmogorov_smirnov, local_mixing_ratio, gromov_wasserstein, internal_geometry

if __name__ == "__main__":
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
    save_path = f"images/global_fidelity/"
    path=Path(save_path)
    path.mkdir(parents=True, exist_ok=True)

    #### Data Preparation ####
    # Get Real Data & Holdout Data
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    trimmed_real_fd, real_landmarks_all = extract_ecg_clinical_landmarks(real_all, n_beats, sr)

    n_samples = trimmed_real_fd.data_matrix.shape[0]
    if n_samples < 2 * n_data:
        raise ValueError(
            f"Need at least {2 * n_data} samples, got {n_samples} after landmark extraction."
        )
    sampled_idx = np.random.choice(n_samples, size=2 * n_data, replace=False)
    real_idx = sampled_idx[:n_data]
    holdout_idx = sampled_idx[n_data:]

    real_fd = trimmed_real_fd[real_idx]
    real_landmarks = real_landmarks_all[real_idx]
    holdout_fd = trimmed_real_fd[holdout_idx]
    holdout_landmarks = real_landmarks_all[holdout_idx]

    # Get Synthetic Data
    synthetic_all = load_synthetic_dataset(diagnostic, lead)
    trimmed_synthetic_fd, synthetic_landmarks_all = extract_ecg_clinical_landmarks(synthetic_all, n_beats, sr)
    synthetic_fd = trimmed_synthetic_fd[:n_data]
    synthetic_landmarks = synthetic_landmarks_all[:n_data]

    #### Transformation ####
    # Apply FPCA on holdout dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(holdout_fd, n_basis, domain_range)
    holdout_fd_smooth, _, _, _ = basis_smoothing_with_lambda(holdout_fd, lambda_, n_basis, domain_range)
    holdout_aligned_fd, _ = landmark_registration(holdout_fd_smooth, holdout_landmarks, landmark_locations)
    holdout_fpca_mean, holdout_fpca_components, holdout_fpca_scores, holdout_fpca_var_ratio, holdout_fpca_ = fpca_with_param(holdout_aligned_fd, n_components)

    # Apply Holdout FPCA on Real & Synthetic
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks, landmark_locations)
    real_scores = holdout_fpca_.transform(real_aligned_fd)

    synthetic_fd_smooth, _, _, _ = basis_smoothing_with_lambda(synthetic_fd, lambda_, n_basis, domain_range)
    synthetic_aligned_fd, _ = landmark_registration(synthetic_fd_smooth, synthetic_landmarks, landmark_locations)
    synthetic_scores = holdout_fpca_.transform(synthetic_fd)
    
    # Apply Isomap on holdout, real, and synthetic FPC scores separately
    optimal_k_holdout = find_optimal_k(holdout_fpca_scores)
    optimal_dim_holdout = find_optimal_manifold_dim(holdout_fpca_scores, optimal_k_holdout)
    isomap_holdout = Isomap(n_neighbors=optimal_k_holdout, n_components=optimal_dim_holdout)
    holdout_embedding = isomap_holdout.fit_transform(holdout_fpca_scores)

    optimal_k_real = find_optimal_k(real_scores)
    optimal_dim_real = find_optimal_manifold_dim(real_scores, optimal_k_real)
    isomap_real = Isomap(n_neighbors=optimal_k_real, n_components=optimal_dim_real)
    real_embedding = isomap_real.fit_transform(real_scores)

    optimal_k_synthetic = find_optimal_k(synthetic_scores)
    optimal_dim_synthetic = find_optimal_manifold_dim(synthetic_scores, optimal_k_synthetic)
    isomap_synthetic = Isomap(n_neighbors=optimal_k_synthetic, n_components=optimal_dim_synthetic)
    synthetic_embedding = isomap_synthetic.fit_transform(synthetic_scores)

    #### ------ Fidelity Evaluation ------ ####
    #### Training Gap
    ## Linear Fidelity with FPC scores
    tg_fpc_mmd = mmd_distance(holdout_fpca_scores, real_scores)
    tg_fpc_frechet_wasserstein = frechet_wasserstein(holdout_fpca_scores, real_scores)
    tg_fpc_covariance_dist = covariance_operator_dist(holdout_fpca_scores, real_scores)
    tg_fpc_kolmogorov_smirnov = kolmogorov_smirnov(holdout_fpca_scores, real_scores)
    tg_fpc_precisions, tg_fpc_recalls, tg_fpc_densities, tg_fpc_coverages, tg_fpc_prdc_ks = compute_prdc(holdout_fpca_scores, real_scores)
    tg_fpc_ratios, tg_fpc_baseline, tg_fpc_ratio_ks = local_mixing_ratio(holdout_fpca_scores, real_scores)

    ## Non-Linear Fidelity with Isomap embeddings
    tg_iso_gromov_wasserstein = gromov_wasserstein(holdout_embedding, real_embedding)
    tg_iso_internal_geometry = internal_geometry(isomap_holdout.dist_matrix_, isomap_real.dist_matrix_)

    #### Synthetic Gap
    sg_fpc_mmd = mmd_distance(holdout_fpca_scores, synthetic_scores)
    sg_fpc_frechet_wasserstein = frechet_wasserstein(holdout_fpca_scores, synthetic_scores)
    sg_fpc_covariance_dist = covariance_operator_dist(holdout_fpca_scores, synthetic_scores)
    sg_fpc_kolmogorov_smirnov = kolmogorov_smirnov(holdout_fpca_scores, synthetic_scores)
    sg_fpc_precisions, sg_fpc_recalls, sg_fpc_densities, sg_fpc_coverages, sg_fpc_prdc_ks = compute_prdc(holdout_fpca_scores, synthetic_scores)
    sg_fpc_ratios, sg_fpc_baseline, sg_fpc_ratio_ks = local_mixing_ratio(holdout_fpca_scores, synthetic_scores)

    sg_iso_gromov_wasserstein = gromov_wasserstein(holdout_embedding, synthetic_embedding)
    sg_iso_internal_geometry = internal_geometry(isomap_holdout.dist_matrix_, isomap_synthetic.dist_matrix_)

    #### Real vs. Synthetic
    rs_fpc_mmd = mmd_distance(real_scores, synthetic_scores)
    rs_fpc_frechet_wasserstein = frechet_wasserstein(real_scores, synthetic_scores)
    rs_fpc_covariance_dist = covariance_operator_dist(real_scores, synthetic_scores)
    rs_fpc_kolmogorov_smirnov = kolmogorov_smirnov(real_scores, synthetic_scores)
    rs_fpc_precisions, rs_fpc_recalls, rs_fpc_densities, rs_fpc_coverages, rs_fpc_prdc_ks = compute_prdc(real_scores, synthetic_scores)
    rs_fpc_ratios, rs_fpc_baseline, rs_fpc_ratio_ks = local_mixing_ratio(real_scores, synthetic_scores)

    rs_iso_gromov_wasserstein = gromov_wasserstein(real_embedding, synthetic_embedding)
    rs_iso_internal_geometry = internal_geometry(isomap_real.dist_matrix_, isomap_synthetic.dist_matrix_)

    #### ------ Results ------ ####
    #### FPCA Plots
    for i, c in enumerate(holdout_fpca_components):
        c.plot()
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"Holdout FPC {i+1}")
        plt.savefig(save_path + f"holdout_fpc_{i+1}.png")
        plt.close()
    holdout_aligned_fd.plot()
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title("Smoothed Aligned Holdout Signal")
    plt.savefig(save_path + "smoothed_holdout.png")
    plt.close()
    real_aligned_fd.plot()
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title("Smoothed Aligned Real Signal")
    plt.savefig(save_path + "smoothed_real.png")
    plt.close()
    synthetic_aligned_fd.plot()
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title("Smoothed Aligned Synthetic Signal")
    plt.savefig(save_path + "smoothed_synthetic.png")
    plt.close()

    print(f"Holdout FPC Variance Ratio: {np.sum(holdout_fpca_var_ratio)}")

    #### Isomap Hyperparameter Settings
    print(f"Holdout Isomap Optimal k: {optimal_k_holdout}, Optimal dim: {optimal_dim_holdout}")
    print(f"Real Isomap Optimal k: {optimal_k_real}, Optimal dim: {optimal_dim_real}")
    print(f"Synthetic Isomap Optimal k: {optimal_k_synthetic}, Optimal dim: {optimal_dim_synthetic}")
    
    #### Tables
    column = ["Metric", "Training Gap", "Synthetic Gap", "Real vs. Synthetic"]
    table = [
        ["MMD", tg_fpc_mmd, sg_fpc_mmd, rs_fpc_mmd],
        ["Frechet Wasserstein", tg_fpc_frechet_wasserstein, sg_fpc_frechet_wasserstein, rs_fpc_frechet_wasserstein],
        ["Covariance Distance", tg_fpc_covariance_dist, sg_fpc_covariance_dist, rs_fpc_covariance_dist],
        ["Gromov Wasserstein", tg_iso_gromov_wasserstein, sg_iso_gromov_wasserstein, rs_iso_gromov_wasserstein],
        ["Internal Geometry", tg_iso_internal_geometry, sg_iso_internal_geometry, rs_iso_internal_geometry],
    ]
    print(tb.tabulate(table, headers=column, tablefmt="grid"))

    column = ["Kolmogorov Smirnov", "Training Gap", "Synthetic Gap", "Real vs. Synthetic"]
    table = [
        [f"FPC {i+1}", tg_fpc_kolmogorov_smirnov[i], sg_fpc_kolmogorov_smirnov[i], rs_fpc_kolmogorov_smirnov[i]] for i in range(n_components)
    ]
    print(tb.tabulate(table, headers=column, tablefmt="grid"))

    #### Plotting
    ## Training Gap
    plt.plot(tg_fpc_prdc_ks, tg_fpc_precisions, label="Precision")
    plt.plot(tg_fpc_prdc_ks, tg_fpc_recalls, label="Recall")
    plt.plot(tg_fpc_prdc_ks, tg_fpc_densities, label="Density")
    plt.plot(tg_fpc_prdc_ks, tg_fpc_coverages, label="Coverage")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Score")
    plt.title("Training Gap PRDC")
    plt.legend()
    plt.savefig(save_path + "Linear_Fidelity_PRDC_tg.png")
    plt.close()

    plt.plot(tg_fpc_ratio_ks, tg_fpc_ratios, label="Ratio")
    plt.axhline(y=tg_fpc_baseline, color='red', linestyle='--', label="Baseline")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Local Mixing Ratio")
    plt.title("Training Gap Local Mixing Ratio")
    plt.legend()
    plt.savefig(save_path + "Linear_Fidelity_Ratio_tg.png")
    plt.close()

    ## Synthetic Gap
    plt.plot(sg_fpc_prdc_ks, sg_fpc_precisions, label="Precision")
    plt.plot(sg_fpc_prdc_ks, sg_fpc_recalls, label="Recall")
    plt.plot(sg_fpc_prdc_ks, sg_fpc_densities, label="Density")
    plt.plot(sg_fpc_prdc_ks, sg_fpc_coverages, label="Coverage")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Score")
    plt.title("Synthetic Gap PRDC")
    plt.legend()
    plt.savefig(save_path + "Linear_Fidelity_PRDC_sg.png")
    plt.close()

    plt.plot(sg_fpc_ratio_ks, sg_fpc_ratios, label="Ratio")
    plt.axhline(y=sg_fpc_baseline, color='red', linestyle='--', label="Baseline")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Local Mixing Ratio")
    plt.title("Synthetic Gap Local Mixing Ratio")
    plt.legend()
    plt.savefig(save_path + "Linear_Fidelity_Ratio_sg.png")
    plt.close()

    ## Real vs. Synthetic
    plt.plot(rs_fpc_prdc_ks, rs_fpc_precisions, label="Precision")
    plt.plot(rs_fpc_prdc_ks, rs_fpc_recalls, label="Recall")
    plt.plot(rs_fpc_prdc_ks, rs_fpc_densities, label="Density")
    plt.plot(rs_fpc_prdc_ks, rs_fpc_coverages, label="Coverage")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Score")
    plt.title("Real vs. Synthetic PRDC")
    plt.legend()
    plt.savefig(save_path + "Linear_Fidelity_PRDC_rs.png")
    plt.close()

    plt.plot(rs_fpc_ratio_ks, rs_fpc_ratios, label="Ratio")
    plt.axhline(y=rs_fpc_baseline, color='red', linestyle='--', label="Baseline")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Local Mixing Ratio")
    plt.title("Real vs. Synthetic Local Mixing Ratio")
    plt.legend()
    plt.savefig(save_path + "Linear_Fidelity_Ratio_rs.png")
    plt.close()
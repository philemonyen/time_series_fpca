import numpy as np
import tabulate as tb
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from pathlib import Path
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks, load_synthetic_dataset
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.transformation.fpca import fpca_with_param
from methods.transformation.isomap import find_optimal_k, find_optimal_manifold_dim
from methods.evaluation.fidelity import mmd_distance, frechet_wasserstein, covariance_operator_dist, compute_prdc, kolmogorov_smirnov, local_mixing_ratio, gromov_wasserstein, internal_geometry

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
    np.random.seed(42)

    #### Data Preparation ####
    # Get Real Data
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    trimmed_real_fd, real_landmarks_all = extract_ecg_clinical_landmarks(real_all, n_beats, sr)

    # Get Synthetic Data
    synthetic_all = load_synthetic_dataset(diagnostic, lead)
    trimmed_synthetic_fd, synthetic_landmarks_all = extract_ecg_clinical_landmarks(synthetic_all, n_beats, sr)
    synthetic_fd = trimmed_synthetic_fd[:n_data]
    synthetic_landmarks = synthetic_landmarks_all[:n_data]

    #### Transformation ####
    # Apply FPCA on Real dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(trimmed_real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(trimmed_real_fd, lambda_, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks_all, landmark_locations)
    real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(real_aligned_fd, n_components)

    # Apply Real FPCA on Synthetic
    synthetic_fd_smooth, _, _, _ = basis_smoothing_with_lambda(trimmed_synthetic_fd, lambda_, n_basis, domain_range)
    synthetic_aligned_fd, _ = landmark_registration(synthetic_fd_smooth, synthetic_landmarks_all, landmark_locations)
    synthetic_scores = real_fpca_.transform(synthetic_aligned_fd)
    
    # Apply Isomap on real and synthetic FPC scores separately
    optimal_k_real = find_optimal_k(real_scores)
    optimal_dim_real = find_optimal_manifold_dim(real_scores, optimal_k_real)
    isomap_real = Isomap(n_neighbors=optimal_k_real, n_components=optimal_dim_real)
    real_embedding = isomap_real.fit_transform(real_scores)

    optimal_k_synthetic = find_optimal_k(synthetic_scores)
    optimal_dim_synthetic = find_optimal_manifold_dim(synthetic_scores, optimal_k_synthetic)
    isomap_synthetic = Isomap(n_neighbors=optimal_k_synthetic, n_components=optimal_dim_synthetic)
    synthetic_embedding = isomap_synthetic.fit_transform(synthetic_scores)

    #### ------ Fidelity Evaluation ------ ####
    mmd = mmd_distance(real_scores, synthetic_scores)
    fw = frechet_wasserstein(real_scores, synthetic_scores)
    covariance_dist = covariance_operator_dist(real_scores, synthetic_scores)
    ks_test = kolmogorov_smirnov(real_scores, synthetic_scores)
    precisions, recalls, densities, coverages, prdc_ks = compute_prdc(real_scores, synthetic_scores)
    ratios, baseline, ratio_ks = local_mixing_ratio(real_scores, synthetic_scores)

    gw= gromov_wasserstein(real_embedding, synthetic_embedding)
    # ig = internal_geometry(isomap_real.dist_matrix_, isomap_synthetic.dist_matrix_)

    #### ------ Results ------ ####
    #### FPCA Plots
    for i, c in enumerate(real_components):
        c.plot()
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"Real FPC {i+1}")
        plt.savefig(save_path + f"real_fpc_{i+1}.png")
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

    print(f"Real FPC Variance Ratio: {np.sum(real_var_ratio)}")

    #### Isomap Hyperparameter Settings
    print(f"Real Isomap Optimal k: {optimal_k_real}, Optimal dim: {optimal_dim_real}")
    print(f"Synthetic Isomap Optimal k: {optimal_k_synthetic}, Optimal dim: {optimal_dim_synthetic}")
    
    #### Tables
    column = ["Metric", "Score"]
    table = [
        ["MMD", mmd], 
        ["Frechet Wasserstein", fw], 
        ["Covariance Distance", covariance_dist], 
        ["Gromov Wasserstein", gw],
        # ["Internal Geometry", ig],
    ]
    print(tb.tabulate(table, headers=column, tablefmt="grid"))

    column = ["Kolmogorov Smirnov", "Score"]
    table = [
        [f"FPC {i+1}", ks_test[i]] for i in range(n_components)
    ]
    print(tb.tabulate(table, headers=column, tablefmt="grid"))

    #### Plotting
    plt.plot(prdc_ks, precisions, label="Precision")
    plt.plot(prdc_ks, recalls, label="Recall")
    plt.plot(prdc_ks, densities, label="Density")
    plt.plot(prdc_ks, coverages, label="Coverage")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Score")
    plt.title("PRDC")
    plt.legend()
    plt.savefig(save_path + "PRDC.png")
    plt.close()

    plt.plot(ratio_ks, ratios, label="Ratio")
    plt.axhline(y=baseline, color='red', linestyle='--', label="Baseline")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Local Mixing Ratio")
    plt.title("Local Mixing Ratio")
    plt.legend()
    plt.savefig(save_path + "Local_Mixing_Ratio.png")
    plt.close()
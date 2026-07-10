import numpy as np
import tabulate as tb
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from pathlib import Path
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.transformation.fda.fpca import fpca_with_param
from methods.transformation.nonlinear.isomap import find_optimal_k, find_optimal_manifold_dim
from methods.evaluation.fidelity import mmd_distance, frechet_wasserstein, covariance_operator_dist, compute_prdc, kolmogorov_smirnov, local_mixing_ratio, gromov_wasserstein, internal_geometry
from methods.validation.controlled_flaw_modelling import gaussian_noise, full_memorization, segment_leaking, mode_collapse

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
    save_path = f"images/fidelity_validation/"
    path=Path(save_path)
    path.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    #### Data Preparation ####
    ## Real Data
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    trimmed_real_fd, real_landmarks_all = extract_ecg_clinical_landmarks(real_all, n_beats, sr)

    # Apply FPCA on Real dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(trimmed_real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(trimmed_real_fd, lambda_, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks_all, landmark_locations)
    real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(real_aligned_fd, n_components)
    
    # Apply Isomap on real and validation FPC scores separately
    optimal_k_real = find_optimal_k(real_scores)
    optimal_dim_real = find_optimal_manifold_dim(real_scores, optimal_k_real)
    isomap_real = Isomap(n_neighbors=optimal_k_real, n_components=optimal_dim_real)
    real_embedding = isomap_real.fit_transform(real_scores)

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
    print(f"Real FPC Variance Ratio: {np.sum(real_var_ratio)}")
    print(f"Real Isomap Optimal k: {optimal_k_real}, Optimal dim: {optimal_dim_real}")
    
    ## Poor Fidelity Validation
    mmd, fw, cd, gw, ks = [], [], [], [], []
    mult_grid = [1.0, 1.5, 2.0, 2.5, 3.0]
    for mult in mult_grid:
        path=Path(save_path + f"low_fidelity_{mult}/")
        path.mkdir(parents=True, exist_ok=True)

        low_fidelity_dataset = gaussian_noise(trimmed_real_fd, mult)
        low_fidelity_fd_smooth, _, _, _ = basis_smoothing_with_lambda(low_fidelity_dataset, lambda_, n_basis, domain_range)
        low_fidelity_aligned_fd, _ = landmark_registration(low_fidelity_fd_smooth, real_landmarks_all, landmark_locations)
        low_fidelity_scores = real_fpca_.transform(low_fidelity_aligned_fd)

        optimal_k_low_fidelity = find_optimal_k(low_fidelity_scores)
        optimal_dim_low_fidelity = find_optimal_manifold_dim(low_fidelity_scores, optimal_k_low_fidelity)
        isomap_low_fidelity = Isomap(n_neighbors=optimal_k_low_fidelity, n_components=optimal_dim_low_fidelity)
        low_fidelity_embedding = isomap_low_fidelity.fit_transform(low_fidelity_scores)
        
        mmd_low_fidelity = mmd_distance(real_scores, low_fidelity_scores)
        fw_low_fidelity = frechet_wasserstein(real_scores, low_fidelity_scores)
        covariance_dist_low_fidelity = covariance_operator_dist(real_scores, low_fidelity_scores)
        ks_test_low_fidelity = kolmogorov_smirnov(real_scores, low_fidelity_scores)
        precisions_low_fidelity, recalls_low_fidelity, densities_low_fidelity, coverages_low_fidelity, prdc_ks_low_fidelity = compute_prdc(real_scores, low_fidelity_scores)
        ratios_low_fidelity, baseline_low_fidelity, ratio_ks_low_fidelity = local_mixing_ratio(real_scores, low_fidelity_scores)
        
        gw_low_fidelity = gromov_wasserstein(real_embedding, low_fidelity_embedding)

        mmd.append(mmd_low_fidelity)
        fw.append(fw_low_fidelity)
        cd.append(covariance_dist_low_fidelity)
        gw.append(gw_low_fidelity)
        ks.append(ks_test_low_fidelity)

        low_fidelity_aligned_fd.plot()
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"Smoothed Aligned Low Fidelity Signal: {mult} Multiplier")
        plt.savefig(path / "smoothed_low_fidelity.png")
        plt.close()
        
        plt.plot(prdc_ks_low_fidelity, precisions_low_fidelity, label="Precision")
        plt.plot(prdc_ks_low_fidelity, recalls_low_fidelity, label="Recall")
        plt.plot(prdc_ks_low_fidelity, densities_low_fidelity, label="Density")
        plt.plot(prdc_ks_low_fidelity, coverages_low_fidelity, label="Coverage")
        plt.xlabel("Number of Neighbors")
        plt.ylabel("Score")
        plt.title(f"PRDC: {mult} Multiplier")
        plt.legend()
        plt.savefig(path / "PRDC_low_fidelity.png")
        plt.close()

        plt.plot(ratio_ks_low_fidelity, ratios_low_fidelity, label="Ratio")
        plt.axhline(y=baseline_low_fidelity, color='red', linestyle='--', label="Baseline")
        plt.xlabel("Number of Neighbors")
        plt.ylabel("Local Mixing Ratio")
        plt.title(f"Local Mixing Ratio: {mult} Multiplier")
        plt.legend()
        plt.savefig(path / "Local_Mixing_Ratio_low_fidelity.png")
        plt.close()

    column = ["Metric", *mult_grid]
    table = [
        ["MMD", *mmd], 
        ["Frechet Wasserstein", *fw], 
        ["Covariance Distance", *cd], 
        ["Gromov Wasserstein", *gw],
    ]
    print(f"Poor Fidelity Results")
    print(tb.tabulate(table, headers=column, tablefmt="grid"))

    ks = np.array(ks)
    column = ["Kolmogorov Smirnov"] + [f"FPC {i+1}" for i in range(n_components)]
    table = [
        [f"Mult {mult}", *ks[i, :].tolist()] for i, mult in enumerate(mult_grid) 
    ]
    print(tb.tabulate(table, headers=column, tablefmt="grid"))

    ## Mode Collapse Dataset
    mmd, fw, cd, gw, ks = [], [], [], [], []
    num_mode_grid = [1, 2, 3, 4, 5]
    for num_modes in num_mode_grid:
        path=Path(save_path + f"mode_collapse_{num_modes}/")
        path.mkdir(parents=True, exist_ok=True)

        mode_collapse_dataset, mode_collapse_landmarks = mode_collapse(trimmed_real_fd, real_landmarks_all, num_modes=num_modes)
        mode_collapse_fd_smooth, _, _, _ = basis_smoothing_with_lambda(mode_collapse_dataset, lambda_, n_basis, domain_range)
        mode_collapse_aligned_fd, _ = landmark_registration(mode_collapse_fd_smooth, mode_collapse_landmarks, landmark_locations)
        mode_collapse_scores = real_fpca_.transform(mode_collapse_aligned_fd)

        optimal_k_mode_collapse = find_optimal_k(mode_collapse_scores, max_k=100)
        optimal_dim_mode_collapse = find_optimal_manifold_dim(mode_collapse_scores, optimal_k_mode_collapse)
        isomap_mode_collapse = Isomap(n_neighbors=optimal_k_mode_collapse, n_components=optimal_dim_mode_collapse)
        mode_collapse_embedding = isomap_mode_collapse.fit_transform(mode_collapse_scores)
    
        mmd_mode_collapse = mmd_distance(real_scores, mode_collapse_scores)
        fw_mode_collapse = frechet_wasserstein(real_scores, mode_collapse_scores)
        covariance_dist_mode_collapse = covariance_operator_dist(real_scores, mode_collapse_scores)
        ks_test_mode_collapse = kolmogorov_smirnov(real_scores, mode_collapse_scores)
        precisions_mode_collapse, recalls_mode_collapse, densities_mode_collapse, coverages_mode_collapse, prdc_ks_mode_collapse = compute_prdc(real_scores, mode_collapse_scores)
        ratios_mode_collapse, baseline_mode_collapse, ratio_ks_mode_collapse = local_mixing_ratio(real_scores, mode_collapse_scores)

        gw_mode_collapse = gromov_wasserstein(real_embedding, mode_collapse_embedding)

        mmd.append(mmd_mode_collapse)
        fw.append(fw_mode_collapse)
        cd.append(covariance_dist_mode_collapse)
        gw.append(gw_mode_collapse)
        ks.append(ks_test_mode_collapse)

        mode_collapse_aligned_fd.plot()
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"Smoothed Aligned Mode Collapse Signal: {num_modes} Modes")
        plt.savefig(path / "smoothed_mode_collapse.png")
        plt.close()
        
        plt.plot(prdc_ks_mode_collapse, precisions_mode_collapse, label="Precision")
        plt.plot(prdc_ks_mode_collapse, recalls_mode_collapse, label="Recall")
        plt.plot(prdc_ks_mode_collapse, densities_mode_collapse, label="Density")
        plt.plot(prdc_ks_mode_collapse, coverages_mode_collapse, label="Coverage")
        plt.xlabel("Number of Neighbors")
        plt.ylabel("Score")
        plt.title(f"PRDC: {num_modes} Modes")
        plt.legend()
        plt.savefig(path / "PRDC_mode_collapse.png")
        plt.close()

        plt.plot(ratio_ks_mode_collapse, ratios_mode_collapse, label="Ratio")
        plt.axhline(y=baseline_mode_collapse, color='red', linestyle='--', label="Baseline")
        plt.xlabel("Number of Neighbors")
        plt.ylabel("Local Mixing Ratio")
        plt.title(f"Local Mixing Ratio: {num_modes} Modes")
        plt.legend()
        plt.savefig(path / "Local_Mixing_Ratio_mode_collapse.png")
        plt.close()

    column = ["Metric", *num_mode_grid]
    table = [
        ["MMD", *mmd], 
        ["Frechet Wasserstein", *fw], 
        ["Covariance Distance", *cd], 
        ["Gromov Wasserstein", *gw],
    ]
    print("Mode Collapse Results")
    print(tb.tabulate(table, headers=column, tablefmt="grid"))

    ks = np.array(ks)
    column = ["Kolmogorov Smirnov"] + [f"FPC {i+1}" for i in range(n_components)]
    table = [
        [f"Num Modes {num_mode}", *ks[i, :].tolist()] for i, num_mode in enumerate(num_mode_grid) 
    ]
    print(tb.tabulate(table, headers=column, tablefmt="grid"))

    ## Exact Memorization Dataset
    mmd, fw, cd, gw, ks = [], [], [], [], []
    num_memorized_grid = [50, 100, 150, 200, 250]
    for num_memorized in num_memorized_grid:
        path=Path(save_path + f"exact_memorization_{num_memorized}/")
        path.mkdir(parents=True, exist_ok=True)

        exact_memorization_dataset, exact_memorization_landmarks = full_memorization(trimmed_real_fd, real_landmarks_all)
        exact_memorization_fd_smooth, _, _, _ = basis_smoothing_with_lambda(exact_memorization_dataset, lambda_, n_basis, domain_range)
        exact_memorization_aligned_fd, _ = landmark_registration(exact_memorization_fd_smooth, exact_memorization_landmarks, landmark_locations)
        exact_memorization_scores = real_fpca_.transform(exact_memorization_aligned_fd)
        
        optimal_k_exact_memorization = find_optimal_k(exact_memorization_scores)
        optimal_dim_exact_memorization = find_optimal_manifold_dim(exact_memorization_scores, optimal_k_exact_memorization)
        isomap_exact_memorization = Isomap(n_neighbors=optimal_k_exact_memorization, n_components=optimal_dim_exact_memorization)
        exact_memorization_embedding = isomap_exact_memorization.fit_transform(exact_memorization_scores)
        mmd_exact_memorization = mmd_distance(real_scores, exact_memorization_scores)
        fw_exact_memorization = frechet_wasserstein(real_scores, exact_memorization_scores)
        covariance_dist_exact_memorization = covariance_operator_dist(real_scores, exact_memorization_scores)
        ks_test_exact_memorization = kolmogorov_smirnov(real_scores, exact_memorization_scores)
        precisions_exact_memorization, recalls_exact_memorization, densities_exact_memorization, coverages_exact_memorization, prdc_ks_exact_memorization = compute_prdc(real_scores, exact_memorization_scores)
        ratios_exact_memorization, baseline_exact_memorization, ratio_ks_exact_memorization = local_mixing_ratio(real_scores, exact_memorization_scores)

        gw_exact_memorization = gromov_wasserstein(real_embedding, exact_memorization_embedding)

        mmd.append(mmd_exact_memorization)
        fw.append(fw_exact_memorization)
        cd.append(covariance_dist_exact_memorization)
        gw.append(gw_exact_memorization)
        ks.append(ks_test_exact_memorization)

        exact_memorization_aligned_fd.plot()
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"Smoothed Aligned Exact Memorization Signal: {num_memorized} Memorized")
        plt.savefig(path / "smoothed_exact_memorization.png")
        plt.close()
        
        plt.plot(prdc_ks_exact_memorization, precisions_exact_memorization, label="Precision")
        plt.plot(prdc_ks_exact_memorization, recalls_exact_memorization, label="Recall")
        plt.plot(prdc_ks_exact_memorization, densities_exact_memorization, label="Density")
        plt.plot(prdc_ks_exact_memorization, coverages_exact_memorization, label="Coverage")
        plt.xlabel("Number of Neighbors")
        plt.ylabel("Score")
        plt.title(f"PRDC: {num_memorized} Memorized")
        plt.legend()
        plt.savefig(path / "PRDC_exact_memorization.png")
        plt.close()

        plt.plot(ratio_ks_exact_memorization, ratios_exact_memorization, label="Ratio")
        plt.axhline(y=baseline_exact_memorization, color='red', linestyle='--', label="Baseline")
        plt.xlabel("Number of Neighbors")
        plt.ylabel("Local Mixing Ratio")
        plt.title(f"Local Mixing Ratio: {num_memorized} Memorized")
        plt.legend()
        plt.savefig(path / "Local_Mixing_Ratio_exact_memorization.png")
        plt.close()
    
    column = ["Metric", *num_memorized_grid]
    table = [
        ["MMD", *mmd], 
        ["Frechet Wasserstein", *fw], 
        ["Covariance Distance", *cd], 
        ["Gromov Wasserstein", *gw],
    ]
    print("Exact Memorization Results")
    print(tb.tabulate(table, headers=column, tablefmt="grid"))

    ks = np.array(ks)
    column = ["Kolmogorov Smirnov"] + [f"FPC {i+1}" for i in range(n_components)]
    table = [
        [f"Num Memorized {num_memorized}", *ks[i, :].tolist()] for i, num_memorized in enumerate(num_memorized_grid) 
    ]
    print(tb.tabulate(table, headers=column, tablefmt="grid"))
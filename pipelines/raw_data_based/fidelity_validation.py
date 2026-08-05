import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from preprocess.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks
from preprocess.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from metrics.fidelity import mmd, wasserstein, local_mixing_ratio
from transformation.nonlinear.umap import tune_umap
from validation.dataset_creation import *

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
    save_path = f"images/fidelity_val/baseline/"
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
    
    # Registration of Real dataset
    lambda_ = basis_smoothing_hyperparameter_tuning(trimmed_real_fd, n_basis, domain_range)
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(trimmed_real_fd, lambda_, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_fd_smooth, real_landmarks_all, landmark_locations)

    # Create Controlled Flaw Dataset
    scenarios = ["oversmoothing", "memorization", "gaussian_noise", "mode_collapse_vary_modes", "mode_collapse_vary_spike_ratio", "segment_leaking"]
    datasets = {}
    # Result Tracking
    result_tracking = {}
    for scenario in scenarios:
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
            result_tracking[scenario][key] = {}
            flaw_fd, flaw_landmarks = value
            # Registration of Flaw dataset
            lambda_ = basis_smoothing_hyperparameter_tuning(flaw_fd, n_basis, domain_range)
            flaw_fd_smooth, _, _, _ = basis_smoothing_with_lambda(flaw_fd, lambda_, n_basis, domain_range)
            flaw_aligned_fd, _ = landmark_registration(flaw_fd_smooth, flaw_landmarks, landmark_locations)
            mmd_score = mmd(real_aligned_fd.data_matrix.squeeze(), flaw_aligned_fd.data_matrix.squeeze())
            wasserstein_score = wasserstein(real_aligned_fd.data_matrix.squeeze(), flaw_aligned_fd.data_matrix.squeeze())
            result_tracking[scenario][key] = {
                "MMD": mmd_score,
                "Wasserstein": wasserstein_score,
            }
            lmr = local_mixing_ratio(real_aligned_fd.data_matrix.squeeze(), flaw_aligned_fd.data_matrix.squeeze())

            real_umap = tune_umap(real_aligned_fd.data_matrix.squeeze())
            real_umap_embedding = real_umap.transform(real_aligned_fd.data_matrix.squeeze())
            flaw_umap_embedding = real_umap.transform(flaw_aligned_fd.data_matrix.squeeze())

            plt.scatter(real_umap_embedding[:, 0], real_umap_embedding[:, 1], label="Real")
            plt.scatter(flaw_umap_embedding[:, 0], flaw_umap_embedding[:, 1], label="Flaw")
            plt.title(f"UMAP Embedding: {scenario}, Flaw Scale: {key}")
            plt.legend()
            plt.savefig(save_path + f"UMAP_Embedding_{scenario}_{key}.png")
            plt.close()

    # Save Results
    with open(save_path + "result_tracking.json", "w") as f:
        json.dump(result_tracking, f)
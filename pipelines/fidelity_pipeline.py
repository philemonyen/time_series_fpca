import sys
from pathlib import Path
# Project root (parent of experiments/) so `methods` resolves when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import json
from methods.fpca import fpca_with_param
from methods.preprocess import basis_smoothing_with_lambda, landmark_registration
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks, load_synthetic_dataset
from methods.evaluation import euclidean, krzanowski_similarity

if __name__ == "__main__":
    diagnostic = "NORM"
    lead = 1
    n_data = 100
    sr = get_sr()
    n_beats = 10
    domain_range = (0, 1)
    n_timepoints = n_beats * sr
    n_basis = int(n_timepoints / 2)
    n_components = 10

    # Get Real Data
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    trimmed_real_fd, real_landmarks_all = extract_ecg_clinical_landmarks(real_all, n_beats, sr)
    real_fd = trimmed_real_fd[:n_data]
    real_landmarks = real_landmarks_all[:n_data]

    # Get Synthetic Data
    synthetic_all = load_synthetic_dataset(diagnostic, lead)
    trimmed_synthetic_fd, synthetic_landmarks_all = extract_ecg_clinical_landmarks(synthetic_all, n_beats, sr)
    synthetic_fd = trimmed_synthetic_fd[:n_data]
    synthetic_landmarks = synthetic_landmarks_all[:n_data]

    # Apply FPCA on Real and Synthetic
    real_smooth_fd, _, _, _ = basis_smoothing_with_lambda(real_fd, 0, n_basis, domain_range)
    real_aligned_fd, _ = landmark_registration(real_smooth_fd, real_landmarks)
    synthetic_smooth_fd, _, _, _ = basis_smoothing_with_lambda(synthetic_fd, 0, n_basis, domain_range)
    synthetic_aligned_fd, _ = landmark_registration(synthetic_smooth_fd, synthetic_landmarks)

    # Apply FPCA on Real and Synthetic
    real_fpca_mean, real_fpca_components, real_fpca_scores, real_fpca_var_ratio, real_fpca_ = fpca_with_param(real_aligned_fd, n_components)
    synthetic_fpca_mean, synthetic_fpca_components, synthetic_fpca_scores, synthetic_fpca_var_ratio, synthetic_fpca_ = fpca_with_param(synthetic_aligned_fd, n_components)

    l2 = euclidean(real_fpca_mean, synthetic_fpca_mean)
    krzanowski = krzanowski_similarity(real_fpca_components, synthetic_fpca_components)
    print("Real NORM vs Synthetic NORM")
    print(f"L2 distance between real and synthetic mean: {l2}")
    print(f"Krzanowski similarity between real and synthetic: {krzanowski}")

    # Get Synthetic Data of different diagnostic
    synthetic_all = load_synthetic_dataset("MI", lead)
    trimmed_synthetic_fd, synthetic_landmarks_all = extract_ecg_clinical_landmarks(synthetic_all, n_beats, sr)
    synthetic_fd = trimmed_synthetic_fd[:n_data]
    synthetic_landmarks = synthetic_landmarks_all[:n_data]
    synthetic_smooth_fd, _, _, _ = basis_smoothing_with_lambda(synthetic_fd, 0, n_basis, domain_range)
    synthetic_aligned_fd, _ = landmark_registration(synthetic_smooth_fd, synthetic_landmarks)
    synthetic_fpca_mean, synthetic_fpca_components, synthetic_fpca_scores, synthetic_fpca_var_ratio, synthetic_fpca_ = fpca_with_param(synthetic_aligned_fd, n_components)
    l2 = euclidean(real_fpca_mean, synthetic_fpca_mean)
    krzanowski = krzanowski_similarity(real_fpca_components, synthetic_fpca_components)
    print("Real NORM vs Synthetic STTC")
    print(f"L2 distance between real and synthetic mean: {l2}")
    print(f"Krzanowski similarity between real and synthetic: {krzanowski}")
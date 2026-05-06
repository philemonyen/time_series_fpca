"""
Module test for basis smoothing

To run this Hyperparameter Tuning Visualization, ensure to adjust preprocess.basis_smoothing_hyperparameter_tuning
to comment out line 109 and uncomment line 113
"""

import sys
from pathlib import Path
# Project root (parent of experiments/) so `methods` resolves when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from methods.preprocess import basis_smoothing_with_lambda
from methods.utils import get_sr, extract_ecg_clinical_landmarks, load_dataset


if __name__ == "__main__":
    diagnostic = "NORM"
    lead = 1
    n_data = 100
    sr = get_sr()
    n_beats = 10
    domain_range = (0, 1)
    n_timepoints = n_beats * sr
    smoothing_parameter = [0, 1e-10, 1e-9, 1e-8]

    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    trimmed_real_fd, landmarks_all = extract_ecg_clinical_landmarks(real_all, n_beats, sr)

    real_fd = trimmed_real_fd[:n_data]
    test_fd = trimmed_real_fd[n_data:2*n_data]

    max_amp, min_amp = np.max(real_fd.data_matrix.squeeze()), np.min(real_fd.data_matrix.squeeze())

    real_fd[0].plot()
    plt.title("Original Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig("../images/basis_smoothing/original.png")
    plt.close()

    for timepoints_per_basis in [2,3,4,5]:
        for lambda_ in smoothing_parameter:
            n_basis = int(n_timepoints / timepoints_per_basis)
            save_path = f"../images/basis_smoothing/{n_basis}"
            path=Path(save_path)
            path.mkdir(parents=True, exist_ok=True)

            fd_smooth, durbin_watson, gcv, edf = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
            fd_smooth[0].plot()
            plt.title(f"Smoothed Signal with {n_basis} basis functions and lambda {lambda_}")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (mV)")
            plt.savefig(f"../images/basis_smoothing/{n_basis}/lambda_{lambda_}.png")
            plt.close()
            print(f"Timepoints per basis: {timepoints_per_basis}, Lambda: {lambda_}")
            print(f"    Durbin-Watson: {durbin_watson}")
            print(f"    Normalized GCV: {np.sqrt(gcv) / (max_amp - min_amp)}")
            print(f"    EDF: {edf}")

    # Fine-grained hyperparameter tuning
    smoothing_parameter = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]
    for lambda_ in smoothing_parameter:
        n_basis = int(n_timepoints / 2)

        fd_smooth, durbin_watson, gcv, edf = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, domain_range)
        print(f"Lambda: {lambda_}: Durbin-Watson: {durbin_watson}")

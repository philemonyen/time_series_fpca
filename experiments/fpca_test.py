"""
Module test for FPCA

"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import numpy as np
import matplotlib.pyplot as plt
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from methods.fpca import fpca_with_param
from methods.utils import load_dataset, get_sr, extract_ecg_clinical_landmarks

if __name__ == "__main__":
    diagnostic = "NORM"
    lead = 1
    n_data = 1000
    sr = get_sr()
    n_beats = 10
    domain_range = (0, 1)
    n_timepoints = n_beats * sr

    # Extract landmarks
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    trimmed_real_fd, landmarks_all = extract_ecg_clinical_landmarks(real_all, n_beats, sr)

    fd = trimmed_real_fd[:n_data]
    landmarks = landmarks_all[:n_data]

    # Plot original signal
    fd.plot()
    plt.title("Original Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig("../images/fpca/original.png")
    plt.close()

    timepoints_per_basis = 2
    n_basis = int(n_timepoints / timepoints_per_basis)
    save_path = f"../images/fpca/{n_basis}"
    path=Path(save_path)
    path.mkdir(parents=True, exist_ok=True)

    # Basis smoothing
    lambda_ = basis_smoothing_hyperparameter_tuning(fd, n_basis, domain_range)
    print(f"Number of basis functions: {n_basis}, Lambda: {lambda_}")
    fd_smooth, _, _, _ = basis_smoothing_with_lambda(fd, lambda_, n_basis, domain_range)

    # Plot smoothed signal
    fd_smooth.plot()
    plt.title("Smoothed Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig(save_path + "/smoothed.png")
    plt.close()

    # Landmark elastic registration
    fd_aligned, warping_ = landmark_registration(fd_smooth, landmarks)

    # Plot aligned signal
    fd_aligned.plot()
    plt.title("Aligned Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig(save_path + "/aligned_landmark.png")
    plt.close()

    # Apply fpca with a fixed number of components
    n_components = 20
    mean, components, scores, var_ratio, fpca_ = fpca_with_param(fd_aligned, n_components)

    print(f"Variance ratio sum for first 10 components: {np.sum(var_ratio[:10])}")
    print(f"Variance ratio sum over {n_components} components: {np.sum(var_ratio)}")
    # print(f"Number of components required to achieve 90% variance: {np.argmin(np.cumsum(var_ratio) >= 0.9) + 1}")

    plt.plot(var_ratio)
    plt.xlabel("Number of components")
    plt.ylabel("Variance ratio")
    plt.title("Variance ratio vs number of components")
    plt.savefig(save_path + "/var_ratio.png")
    plt.close()

    plt.plot(np.cumsum(var_ratio))
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative variance ratio")
    plt.title("Cumulative variance ratio vs number of components")
    plt.savefig(save_path + "/cumulative_var_ratio.png")
    plt.close()

    # Visual inspection for physical meaningfulness
    mean.plot()
    plt.title("Mean Curve")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig(save_path + "/mean.png")
    plt.close()

    for i in range(10):
        plt.plot(components[i].data_matrix.squeeze())
        plt.xlabel("Time (s)")
        plt.ylabel("Variance Significance")
        plt.title(f"Eigenfunction {i+1}")
        plt.savefig(save_path + f"/eigenfunction_{i+1}.png")
        plt.close()
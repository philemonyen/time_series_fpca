"""
Module test for FPCA

"""
import sys
from pathlib import Path
# Project root (parent of experiments/) so `methods` resolves when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, to_fd, elastic_registration
from methods.fpca import get_ecg_info, fpca_hyperparameter_tuning, fpca_with_param, inverse_fpca
from methods.utils import get_data, trim_ecg, get_sr

if __name__ == "__main__":
    diagnostic = ["NORM"]
    lead = 1
    n_data = 100
    n_beats, domain_range = get_ecg_info()
    n_timepoints = n_beats * get_sr()

    real_all = get_data(diagnostic=diagnostic, lead=lead, holdout=False)
    real = trim_ecg(real_all[n_data:2*n_data], n_beats)
    holdout = trim_ecg(real_all[2*n_data:3*n_data], n_beats)
    fd = to_fd(real, 0, n_beats, "time", "voltage")
    fd_holdout = to_fd(holdout, 0, n_beats, "time", "voltage")

    track = []
    for timepoints_per_basis in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        n_basis = int(n_timepoints / timepoints_per_basis)

        # Basis smoothing
        lambda_ = basis_smoothing_hyperparameter_tuning(fd, n_basis, domain_range)
        fd_smooth = basis_smoothing_with_lambda(fd, lambda_, n_basis, domain_range)

        # Elastic registration
        fd_smooth, warping_, template = elastic_registration(fd_smooth)

        # FPCA hyperparameter tuning
        var_ratio = fpca_hyperparameter_tuning(fd_smooth)
        var_ratio_sum = np.cumsum(var_ratio)
        plt.plot(var_ratio_sum)
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative variance ratio")
        plt.title(f"Cumulative variance ratio vs number of components: {n_basis} basis functions")
        plt.savefig(f"../images/fpca/var_ratio_sum_vs_n_components_{n_basis}.png")
        plt.close()

        kl = KneeLocator(range(1, len(var_ratio) + 1), np.cumsum(var_ratio), curve="concave", direction="increasing")
        track.append((n_basis, kl.knee, var_ratio[:kl.knee]))

        # Reconstruction error validation with holdout set
        mean, components, scores, var_ratio, fpca_ = fpca_with_param(fd, kl.knee)

        lambda_ = basis_smoothing_hyperparameter_tuning(fd_holdout, n_basis, domain_range)
        fd_smooth = basis_smoothing_with_lambda(fd_holdout, lambda_, n_basis, domain_range)
        fd_smooth, warping_, template = elastic_registration(fd_smooth)
        scores = fpca_.transform(fd_smooth)
        reconstructed = inverse_fpca(scores, components, mean, warping_)
        reconstruction_error = np.mean(np.linalg.norm(fd_holdout - reconstructed, axis=1))
        track.append((n_basis, kl.knee, var_ratio[:kl.knee], reconstruction_error))


    for n_basis, n_components, var_ratio, reconstruction_error in track:
        print(f"Number of basis functions: {n_basis}:")
        print(f"    Optimal number of components: {n_components}") 
        print(f"    Variance ratio sum: {np.sum(var_ratio)}")
        print(f"    Reconstruction error: {reconstruction_error}")
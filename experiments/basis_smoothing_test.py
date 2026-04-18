"""
Module test for basis smoothing
"""

from methods.preprocess import basis_smoothing_with_hyperparameter_tuning
from methods.fpca import get_ecg_info, to_fd
from methods.utils import get_data, trim_ecg, get_sr
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    diagnostic = ["NORM"]
    lead = 1
    n_data = 1000
    n_beats, domain_range = get_ecg_info()
    n_timepoints = n_beats * get_sr()

    real_all = get_data(diagnostic=diagnostic, lead=lead, holdout=False)
    real = trim_ecg(real_all[n_data:2*n_data], n_beats)
    
    fd = to_fd(real, 0, n_beats, "time", "voltage")
    
    for timepoints_per_basis in [3,4,5,6]:
        n_basis = int(n_timepoints / timepoints_per_basis)
        best_fd_smooth, best_gcv, best_lambda, gcvs, lambdas = basis_smoothing_with_hyperparameter_tuning(fd, n_basis, domain_range)
        print(f"Number of basis functions: {n_basis}:")
        print(f"    Best GCV: {best_gcv}")
        print(f"    Best Lambda: {best_lambda}")

        save_path = f"images/basis_smoothing"
        path=Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        best_fd_smooth.plot()
        plt.title(f"Number of basis functions: {n_basis}")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.savefig(save_path + f"/basis_smoothing_with_hyperparameter_tuning_{n_basis}.png")
        plt.close()

        plt.plot(lambdas, gcvs)
        plt.xlabel("Lambda")
        plt.ylabel("GCV")
        plt.title(f"GCV vs Lambda ({n_basis} basis functions)")
        plt.savefig(save_path + f"/gcv_vs_lambda_{n_basis}.png")
        plt.close()
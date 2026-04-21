"""
Module test for basis smoothing

To run this script, ensure you adjust preprocess.basis_smoothing_hyperparameter_tuning
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
from kneed import KneeLocator
from skfda.representation import FDataGrid
from methods.preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, to_fd
from methods.fpca import get_ecg_info
from methods.utils import get_data, trim_ecg, get_sr


if __name__ == "__main__":
    diagnostic = ["NORM"]
    lead = 1
    n_data = 1000
    n_beats, domain_range = get_ecg_info()
    n_timepoints = n_beats * get_sr()

    real_all = get_data(diagnostic=diagnostic, lead=lead, holdout=False)
    real = trim_ecg(real_all[n_data:2*n_data], n_beats)
    test = trim_ecg(real_all[2*n_data:3*n_data], n_beats)

    max_amp, min_amp = np.max(real), np.min(real)
    print(f"Max amplitude: {max_amp}, Min amplitude: {min_amp}")
    fd = to_fd(real, 0, n_beats, "time", "voltage")
    
    test_fd = to_fd(test, 0, n_beats, "time", "voltage")
    plt_idx = 0
    test_fd[plt_idx].plot()
    plt.title(f"Test ECG")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig(f"../images/basis_smoothing/raw.png")
    plt.close()

    for timepoints_per_basis in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        n_basis = int(n_timepoints / timepoints_per_basis)
        gcvs, lambdas, durbin_watson_scores, edfs = basis_smoothing_hyperparameter_tuning(fd, n_basis, domain_range)
        # Find the knee point of GCV vs Lambda
        # error_rate = np.sqrt(gcvs) / (max_amp - min_amp)
        log_lambdas = np.log10(lambdas)
        gcv_kl = KneeLocator(log_lambdas, gcvs, curve="convex", direction="increasing")
        gcv_opt_idx = log_lambdas.tolist().index(gcv_kl.knee)

        # Find index of DW with elbow method
        durbin_watson_kl = KneeLocator(log_lambdas, durbin_watson_scores, curve="concave", direction="decreasing")
        durbin_watson_opt_idx_elbow = log_lambdas.tolist().index(durbin_watson_kl.knee)

        # Find index of EDF with elbow method
        edf_kl = KneeLocator(log_lambdas, edfs, curve="concave", direction="decreasing")
        edf_opt_idx = log_lambdas.tolist().index(edf_kl.knee) if edf_kl.knee is not None else 1

        print(f"Number of basis functions: {n_basis}:")
        print(f"    Optimal lambda (GCV error rate): {lambdas[gcv_opt_idx]}")
        print(f"    GCV error rate: {np.sqrt(gcvs[gcv_opt_idx]) / (max_amp - min_amp)}")
        print(f"    Optimal lambda (Durbin-Watson): {lambdas[durbin_watson_opt_idx_elbow]}")
        print(f"    Durbin-Watson: {durbin_watson_scores[durbin_watson_opt_idx_elbow]}")
        print(f"    Optimal lambda (Effective degrees of freedom): {lambdas[edf_opt_idx]}")
        print(f"    Effective degrees of freedom: {edfs[edf_opt_idx]}")


        save_path = f"../images/basis_smoothing/{n_basis}"
        path=Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        plt.plot(lambdas, durbin_watson_scores)
        plt.xlabel("Lambda (log scale)")
        plt.xscale("log")
        plt.ylabel("Durbin-Watson Score")
        plt.title(f"Durbin-Watson Score vs Lambda: {n_basis} basis functions")
        plt.savefig(save_path + "/durbin_watson_score_vs_lambda.png")
        plt.close()

        plt.plot(lambdas, edfs)
        plt.xlabel("Lambda (log scale)")
        plt.xscale("log")
        plt.ylabel("Effective Degrees of Freedom")
        plt.title(f"Effective Degrees of Freedom vs Lambda: {n_basis} basis functions")
        plt.savefig(save_path + "/edf_vs_lambda.png")
        plt.close()

        plt.plot(lambdas, gcvs)
        plt.xlabel("Lambda (log scale)")
        plt.xscale("log")
        plt.ylabel("GCV")
        plt.title(f"GCV vs Lambda: {n_basis} basis functions")
        plt.savefig(save_path + "/gcv_vs_lambda.png")
        plt.close()
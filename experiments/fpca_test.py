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
    for timepoints_per_basis in [2, 3, 4, 5]:
        n_basis = int(n_timepoints / timepoints_per_basis)
        save_path = f"../images/fpca/{n_basis}"
        path=Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        # Basis smoothing
        lambda_ = basis_smoothing_hyperparameter_tuning(fd, n_basis, domain_range)
        fd_smooth, _, _, _ = basis_smoothing_with_lambda(fd, lambda_, n_basis, domain_range)

        # Elastic registration, obtain template
        fd_aligned, warping_, template = elastic_registration(fd_smooth)

        # FPCA hyperparameter tuning
        var_ratio = fpca_hyperparameter_tuning(fd_aligned)

        var_ratio_sum = np.cumsum(var_ratio)
        print(var_ratio)
        print(var_ratio_sum)
        print(np.cumsum(var_ratio) >= 0.9)
        print(np.argmax(np.cumsum(var_ratio) >= 0.9) + 1)
        plt.plot(var_ratio_sum)
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative variance ratio")
        plt.title(f"Cumulative variance ratio vs number of components: {n_basis} basis functions")
        plt.savefig(save_path + "/var_ratio_sum_vs_n_components.png")
        plt.close()

        plt.plot(var_ratio)
        plt.xlabel("Number of components")
        plt.ylabel("Variance ratio")
        plt.title(f"Variance ratio vs number of components: {n_basis} basis functions")
        plt.savefig(save_path + "/var_ratio_vs_n_components.png")
        plt.close()

        kl = KneeLocator(range(1, len(var_ratio) + 1), np.cumsum(var_ratio), curve="concave", direction="increasing", interp_method="polynomial", S=1e-4, online=True)
        # Find the optimal number of components considering elbow point and variance ratio sum
        n_components = np.max([kl.knee, np.argmin(np.cumsum(var_ratio) >= 0.9) + 1])
        # Apply fpca with the optimal number of components
        mean, components, scores, var_ratio, fpca_ = fpca_with_param(fd_aligned, n_components)

        # # Reconstruction error validation with holdout set
        # lambda_ = basis_smoothing_hyperparameter_tuning(fd_holdout, n_basis, domain_range)
        # fd_smooth, _, _, _ = basis_smoothing_with_lambda(fd_holdout, lambda_, n_basis, domain_range)
        # fd_aligned, warping_ = elastic_registration(fd_smooth, template)
        # scores = fpca_.transform(fd_aligned)
        # reconstructed = inverse_fpca(scores, components, mean)
        # reconstruction_error = np.mean(np.linalg.norm(fd_aligned.data_matrix.squeeze() - reconstructed.data_matrix, axis=1))

        # # Roughness of Eigenfunctions: integral ((component''(t))^2 dt)
        # component_roughness = []
        # for component in components:
        #     d2 = component.derivative(order=2).data_matrix.squeeze()
        #     grid_points = np.asarray(component.grid_points[0])
        #     roughness = np.trapz(d2 ** 2, grid_points)
        #     component_roughness.append(roughness)
        
        track.append((n_basis, n_components, var_ratio[:n_components]))
        # Bootstrapping

        # Visual inspection for physical meaningfulness
        
        for i in range(n_components):
            plt.plot(components[i].data_matrix.squeeze())
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (mV)")
            plt.title(f"Eigenfunction {i+1}")
            plt.savefig(save_path + f"/eigenfunction_{i+1}.png")
            plt.close()

    for n_basis, n_components, var_ratio in track:
        print(f"Number of basis functions: {n_basis}:")
        print(f"    Optimal number of components: {n_components}") 
        print(f"    Variance ratio sum: {np.sum(var_ratio)}")
        # print(f"    Reconstruction error: {reconstruction_error}")
        # print(f"    Component roughness: {component_roughness}")
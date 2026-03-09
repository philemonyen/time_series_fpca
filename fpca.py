import skfda
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import get_sr
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.registration import FisherRaoElasticRegistration

# Hyperparameter setting
n_beats = 8
n_basis = 500
n_components = 4
domain_range = (0, n_beats)

# ---- FPCA ---- #
class FPCAOutput:
    def __init__(self, fd, 
            smoothed,
            aligned,
            warping,
            template,
            mean,
            components,
            scores,
            var_ratio,
            fpca_):
        self.fd = fd
        self.smoothed = smoothed
        self.aligned = aligned
        self.warping = warping
        self.template = template
        self.mean = mean
        self.components = components
        self.scores = scores
        self.var_ratio = var_ratio
        self.fpca_ = fpca_
        
    def plot(self, name, directory):
        save_path = f"images/{directory}"
        path=Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        self.fd.plot()
        plt.title(f"{name}: Raw ({n_beats} beats)")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.savefig(save_path + '/raw.png')
        plt.close()
        self.smoothed.plot()
        plt.title(f"{name}: Smoothed ({n_beats} beats)")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.savefig(save_path + "/smoothed.png")
        plt.close()
        self.aligned.plot()
        plt.title(f"{name}: Aligned ({n_beats} beats)")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.savefig(save_path + "/aligned.png")
        plt.close()
        self.mean.plot()
        plt.title(f"{name}: FPCA Mean Curve ({n_beats} beats)")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.savefig(save_path + "/mean.png")
        plt.close()
        component_matrix = self.components.data_matrix
        fig, axes = plt.subplots(n_components, 1, figsize=(8, 12))
        xvals = np.linspace(0, n_beats, n_beats*get_sr())
        for i in range(n_components):
            axes[i].plot(xvals, component_matrix[i])
            axes[i].set_title(f"{name}: Eigenfunction {i+1} ({n_beats} beats)")
            axes[i].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(save_path + "/components.png")
        plt.close()

def get_hyperparameters():
    return n_beats, n_basis, n_components, domain_range

def to_fd(data, time_start, time_end, x_axis, y_axis):
    _, seq_len = data.shape
    timepoints = np.linspace(time_start, time_end, seq_len)
    fd = skfda.FDataGrid(
        data_matrix=data,
        grid_points=timepoints,
        argument_names=[x_axis],
        coordinate_names=[y_axis]
    )
    return fd

def basis_smoothing(fd, n_basis, domain_range):
    basis = BSplineBasis(
        n_basis=n_basis,
        domain_range=domain_range,
        order=4
    )
    smoother = BasisSmoother(basis=basis, smoothing_parameter=1e-8)
    fd_smooth = smoother.fit_transform(fd)
    return fd_smooth

def elastic_registration(fd, template=None):
    if template:
        registration = FisherRaoElasticRegistration(template=template)
        fd_aligned = registration.fit_transform(fd)
        warping_ = registration.warping_
        return fd_aligned, warping_
    else:
        registration = FisherRaoElasticRegistration()
        fd_aligned = registration.fit_transform(fd)
        warping_ = registration.warping_
        template_ = registration.template_
        return fd_aligned, warping_, template_

def fpca(fd, n_components):
    fpca_ = FPCA(n_components=n_components)
    scores = fpca_.fit_transform(fd)
    var_ratio = fpca_.explained_variance_ratio_
    mean = fpca_.mean_
    components = fpca_.components_
    return mean, components, scores, var_ratio, fpca_

#--- Inverse FPCA ---- #
def inverse_fpca(scores, components, mean, warping):
    return (scores @ components + mean).transform(warping)

# Pipeline
def fpca_pipeline(data, template_):
    fd = to_fd(data, 0, n_beats, "Time (s)", "Voltage (ms)")
    smoothed = basis_smoothing(fd, n_basis, domain_range)
    if template_:
        aligned, warping = elastic_registration(smoothed, template=template_)
        template = None
    else:
        aligned, warping, template = elastic_registration(smoothed)
    mean, components, scores, var_ratio, fpca_ = fpca(aligned, n_components)
    return FPCAOutput(
        fd, 
        smoothed,
        aligned,
        warping,
        template,
        mean,
        components,
        scores,
        var_ratio,
        fpca_
    )

def fpca_transform_pipeline(fpca_, data):
    fd = to_fd(data, 0, n_beats, "Time (s)", "Voltage (ms)")
    smoothed = basis_smoothing(fd, n_basis, domain_range)
    return fpca_.transform(smoothed)
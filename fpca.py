import skfda
import numpy as np
import matplotlib.pyplot as plt
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.registration import FisherRaoElasticRegistration
from utils import get_data


# ---- FPCA ---- #
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
    fpca = FPCA(n_components=n_components)
    scores = fpca.fit_transform(fd)
    var_ratio = fpca.explained_variance_ratio_
    mean = fpca.mean_
    components = fpca.components_
    return mean, components, scores, var_ratio

#--- Inverse FPCA ---- #
def inverse_fpca(scores, components, mean, warping):
    return (scores @ components + mean).transform(warping)
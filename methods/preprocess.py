from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.registration import FisherRaoElasticRegistration


def basis_smoothing_with_hyperparameter_tuning(fd, n_basis, domain_range):
    """
    Implement basis smoothing hyperparameter tuning using the big-k optimal lambda strategy
    
    Args:
        fd (FDataGrid): the original signal
        n_basis (int): number of basis functions
        domain_range (tuple): (start, end) of the domain
    Returns:
        best_gcv (float): the best GCV score
        best_lambda (float): the best lambda
        best_fd_smooth (FDataGrid): the smoothed signal
    
    Note:
    - n_basis should be large enough to capture the most complex part of the signal
        rule of thumb: 1 basis per 2-5 data points

    """
    basis = BSplineBasis(
        n_basis=n_basis,
        domain_range=domain_range,
        order=4
    )

    # grid search for lambda
    exp_range = [-6, -5, -4, -3, -2, -1, 0, 1, 2]
    gcvs = []
    lambdas = []
    best_gcv = float('inf')
    best_lambda = None
    best_fd_smooth = None
    for exp in exp_range:
        lambda_ = 10**(exp)
        smoother = BasisSmoother(basis=basis, smoothing_parameter=lambda_)
        smoothed = smoother.fit_transform(fd)

        # Calculate GCV
        hat_matrix = smoother.hat_matrix()
        sse = np.sum((fd.data_matrix - smoothed.data_matrix) ** 2)
        n_samples, n_timepoints, n_coordinates = fd.data_matrix.shape
        gcv = (sse / (n_samples * n_timepoints * n_coordinates)) / ((1 - np.trace(hat_matrix) / n_timepoints) ** 2)
        gcvs.append(gcv)
        lambdas.append(lambda_)
        if gcv < best_gcv:
            best_gcv = gcv
            best_lambda = lambda_
            best_fd_smooth = smoothed

    return best_fd_smooth, best_gcv, best_lambda, gcvs, lambdas

def basis_smoothing_with_lambda(fd, lambda_, n_basis, domain_range):
    """
    Implement basis smoothing with a given lambda
    Args:
        fd (FDataGrid): the original signal
        lambda_ (float): the lambda value
        domain_range (tuple): (start, end) of the domain
    Returns:
        fd_smooth (FDataGrid): the smoothed signal
    """
    basis = BSplineBasis(
        n_basis=n_basis,
        domain_range=domain_range,
        order=4
    )
    smoother = BasisSmoother(basis=basis, smoothing_parameter=lambda_)
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
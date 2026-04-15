import numpy as np
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.registration import FisherRaoElasticRegistration


def basis_smoothing(fd, n_basis, domain_range):
    """
    Implement basis smoothing with hyperparameter tuning using the big-k optimal lambda strategy
    
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
    lambda_range = np.logspace(start=-6, stop=2, num=8)
    best_gcv = float('inf')
    best_lambda = None
    best_fd_smooth = None
    for lambda_ in lambda_range:
        smoother = BasisSmoother(basis=basis, smoothing_parameter=lambda_)
        fd_smooth = smoother.fit_transform(fd)
        # calculate the error
        error = np.mean((fd_smooth.data_matrix - fd.data_matrix) ** 2)
        gcv = error / (1 - smoother.n_basis / fd.n_samples) ** 2
        if gcv < best_gcv:
            best_gcv = gcv
            best_lambda = lambda_
            best_fd_smooth = fd_smooth
    return best_gcv, best_lambda, best_fd_smooth

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
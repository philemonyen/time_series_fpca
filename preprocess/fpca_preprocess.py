import skfda
import numpy as np
from kneed import KneeLocator
from skfda.representation import FDataGrid
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.misc.regularization import L2Regularization
from skfda.misc.operators import LinearDifferentialOperator
from skfda.preprocessing.registration import FisherRaoElasticRegistration, landmark_elastic_registration, landmark_elastic_registration_warping

def to_fd(data, time_start, time_end, x_axis, y_axis):
    """
    Transform numpy array (discrete) to FDataGrid (continuous)
    Args:
        data (numpy.ndarray): the data
        time_start (float): the start time
        time_end (float): the end time
        x_axis (str): the x-axis label
        y_axis (str): the y-axis label
    Returns:
        fd (FDataGrid): the FDataGrid
    """
    _, seq_len = data.shape
    timepoints = np.linspace(time_start, time_end, seq_len)
    fd = skfda.FDataGrid(
        data_matrix=data,
        grid_points=timepoints,
        argument_names=[x_axis],
        coordinate_names=[y_axis]
    )
    return fd

def basis_smoothing_hyperparameter_tuning(fd, n_basis, domain_range):
    """
    Implement basis smoothing hyperparameter tuning using the big-k optimal lambda strategy
    
    Args:
        fd (FDataGrid): the original signal
        n_basis (int): number of basis functions
        domain_range (tuple): (start, end) of the domain
    Returns:
        Optimal lambda
    
    Note:
    - n_basis should be large enough to capture the most complex part of the signal
        rule of thumb: 1 basis per 2-5 data points

    """
    basis = BSplineBasis(
        n_basis=n_basis,
        domain_range=domain_range,
        order=4
    )
    penalty = L2Regularization(LinearDifferentialOperator(2))

    # grid search for lambda
    lambdas = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    durbin_watson_scores = []
    for lambda_ in lambdas:
        smoother = BasisSmoother(
            basis=basis, 
            smoothing_parameter=lambda_,
            regularization=penalty)

        smoothed = smoother.fit_transform(fd)
        residual = FDataGrid(data_matrix=fd.data_matrix.squeeze() - smoothed.data_matrix.squeeze())

        # Calculate residual autocorrelation score (Durbin-Watson statistic)
        residual_matrix = residual.data_matrix.squeeze()
        residual_diff = np.diff(residual_matrix, axis=1)
        numerator = np.sum(residual_diff ** 2, axis=1)
        denominator = np.sum(residual_matrix ** 2, axis=1)
        # Mean DW across samples; avoid division by zero for zero-residual curves.
        durbin_watson = np.mean(numerator / np.where(denominator > 0, denominator, 1.0))
        durbin_watson_scores.append(np.abs(durbin_watson - 2))

    lambda_coarse = lambdas[np.argmin(durbin_watson_scores)]
    fine_lambdas = (np.arange(5) + 1) * lambda_coarse
    fine_durbin_watson_scores = []
    for fine_lambda in fine_lambdas:
        smoother = BasisSmoother(
            basis=basis, 
            smoothing_parameter=fine_lambda,
            regularization=penalty)

        smoothed = smoother.fit_transform(fd)
        residual = FDataGrid(data_matrix=fd.data_matrix.squeeze() - smoothed.data_matrix.squeeze())

        residual_matrix = residual.data_matrix.squeeze()
        residual_diff = np.diff(residual_matrix, axis=1)
        numerator = np.sum(residual_diff ** 2, axis=1)
        denominator = np.sum(residual_matrix ** 2, axis=1)
        durbin_watson = np.mean(numerator / np.where(denominator > 0, denominator, 1.0))
        fine_durbin_watson_scores.append(np.abs(durbin_watson - 2))
        
    optimal_lambda = fine_lambdas[np.argmin(fine_durbin_watson_scores)]
    return optimal_lambda

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
    penalty = L2Regularization(LinearDifferentialOperator(2))
    
    smoother = BasisSmoother(
        basis=basis, 
        smoothing_parameter=lambda_,
        regularization=penalty)

    fd_smooth = smoother.fit_transform(fd)

    ### Metric Calculation
    # Durbin-Watson Score
    residual = FDataGrid(data_matrix=fd.data_matrix.squeeze() - fd_smooth.data_matrix.squeeze())
    residual_matrix = residual.data_matrix.squeeze()
    residual_diff = np.diff(residual_matrix, axis=1)
    numerator = np.sum(residual_diff ** 2, axis=1)
    denominator = np.sum(residual_matrix ** 2, axis=1)
    durbin_watson = np.mean(numerator / np.where(denominator > 0, denominator, 1.0))

    # GCV
    hat_matrix = smoother.hat_matrix()
    sse = np.sum((fd.data_matrix - fd_smooth.data_matrix) ** 2)
    n_samples, n_timepoints, n_coordinates = fd.data_matrix.shape
    gcv = (sse / (n_samples * n_timepoints * n_coordinates)) / ((1 - np.trace(hat_matrix) / n_timepoints) ** 2)

    # Effective Degrees of Freedom (EDF)
    edf = np.trace(hat_matrix) / n_timepoints

    
    return fd_smooth, durbin_watson, gcv, edf

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

def landmark_registration(fd, landmarks, locations=None):
    """
    Implement landmark registration
    Args:
        fd (FDataGrid): the original signal
        landmarks (list): the landmarks (P,Q,R,S,T)
    Returns:
        fd_aligned (FDataGrid): the aligned signal
    """
    fd_aligned = landmark_elastic_registration(fd=fd, landmarks=landmarks, location=locations)
    warping_ = landmark_elastic_registration_warping(fd=fd, landmarks=landmarks, location=locations)
    return fd_aligned, warping_
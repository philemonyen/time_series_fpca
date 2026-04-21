import skfda
import numpy as np
from kneed import KneeLocator
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.misc.regularization import L2Regularization
from skfda.misc.operators import LinearDifferentialOperator
from skfda.preprocessing.registration import FisherRaoElasticRegistration


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
    exp_range = [-8, -7, -6, -5, -4, -3, -2]
    gcvs = []
    lambdas = []
    durbin_watson_scores = []
    edfs = []
    for exp in exp_range:
        lambda_ = 10**(exp)
        smoother = BasisSmoother(
            basis=basis, 
            smoothing_parameter=lambda_,
            regularization=penalty)

        smoothed = smoother.fit_transform(fd)
        residual = fd.copy(data_matrix=fd.data_matrix - smoothed.data_matrix)

        # Calculate residual autocorrelation score (Durbin-Watson statistic)
        residual_matrix = residual.data_matrix.squeeze()
        residual_diff = np.diff(residual_matrix, axis=1)
        numerator = np.sum(residual_diff ** 2, axis=1)
        denominator = np.sum(residual_matrix ** 2, axis=1)
        # Mean DW across samples; avoid division by zero for zero-residual curves.
        durbin_watson = np.mean(numerator / np.where(denominator > 0, denominator, 1.0))

        # Calculate GCV
        hat_matrix = smoother.hat_matrix()
        sse = np.sum((fd.data_matrix - smoothed.data_matrix) ** 2)
        n_samples, n_timepoints, n_coordinates = fd.data_matrix.shape
        gcv = (sse / (n_samples * n_timepoints * n_coordinates)) / ((1 - np.trace(hat_matrix) / n_timepoints) ** 2)
        
        # Calculate Effective Degrees of Freedom (EDF)
        ### Looking for 0.1, the general rule of thumb
        edf = np.trace(hat_matrix) / n_timepoints

        gcvs.append(gcv)
        lambdas.append(lambda_)
        durbin_watson_scores.append(durbin_watson)
        edfs.append(edf)


    # Find the optimal lambda
    log_lambdas = np.log10(lambdas)
    gcv_kl = KneeLocator(log_lambdas, gcvs, curve="convex", direction="increasing")
    gcv_opt_idx = log_lambdas.tolist().index(gcv_kl.knee)
    durbin_watson_kl = KneeLocator(log_lambdas, durbin_watson_scores, curve="concave", direction="decreasing")
    durbin_watson_opt_idx_elbow = log_lambdas.tolist().index(durbin_watson_kl.knee)
    edf_kl = KneeLocator(log_lambdas, edfs, curve="concave", direction="decreasing")
    edf_opt_idx = log_lambdas.tolist().index(edf_kl.knee) if edf_kl.knee is not None else 1
    
    optimal_lambda = lambdas[gcv_opt_idx]
    optimal_lambda_elbow = lambdas[durbin_watson_opt_idx_elbow]
    optimal_lambda_edf = lambdas[edf_opt_idx]

    return np.max(optimal_lambda, optimal_lambda_elbow, optimal_lambda_edf)


    ##### For experiment purposes, return the gcvs and lambdas for further analysis
    # return np.array(gcvs), np.array(lambdas), np.array(durbin_watson_scores), np.array(edfs)



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
import numpy as np
from kneed import KneeLocator
from skfda.preprocessing.dim_reduction import FPCA

# Hyperparameter setting
n_beats = 8
domain_range = (0, n_beats)

def get_ecg_info():
    """
    Get ECG information
    Returns:
        n_beats (int): the number of beats
        domain_range (tuple): (start, end) of the domain
    """
    return n_beats, domain_range

def fpca_hyperparameter_tuning(fd):
    """
    Determine optimal number of components using the elbow method, then run FPCA with it
    Args:
        fd (FDataGrid): the original signal
        var_threshold (float): the threshold for the cumulative variance ratio
    Returns:
        Optimal number of components
    """
    # Centering the data by subtracting the mean
    data_matrix = fd.data_matrix
    mean = np.mean(data_matrix, axis=0)
    data_matrix = data_matrix - mean
    fd = fd.copy(data_matrix=data_matrix)

    # Find optimal number of components with elbow method
    max_components = 10
    fpca_ = FPCA(n_components=max_components)
    fpca_.fit(fd)
    var_ratio = fpca_.explained_variance_ratio_
    kl = KneeLocator(np.cumsum(var_ratio), range(1, max_components + 1), curve="convex", direction="increasing")
    
    return kl.knee

def fpca_with_param(fd, n_components):
    """
    Run FPCA with a given hyperparameter

    Args:
        fd (FDataGrid): the original signal
        n_components (int): the number of components
    Returns:
        mean (FDataGrid): the mean curve
        components (FDataGrid): the components
        scores (numpy.ndarray): the scores
        var_ratio (numpy.ndarray): the variance ratio
    """
    # Centering the data by subtracting the mean
    data_matrix = fd.data_matrix
    mean = np.mean(data_matrix, axis=0)
    data_matrix = data_matrix - mean
    fd = fd.copy(data_matrix=data_matrix)

    # FPCA with optimal number of components
    fpca_ = FPCA(n_components=n_components)
    scores = fpca_.fit_transform(fd)
    var_ratio = fpca_.explained_variance_ratio_
    mean = fpca_.mean_
    components = fpca_.components_
    return mean, components, scores, var_ratio

#--- Inverse FPCA ---- #
def inverse_fpca(scores, components, mean, warping):
    return (scores @ components + mean).transform(warping)
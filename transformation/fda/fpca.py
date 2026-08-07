import numpy as np
from skfda.representation import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA

def fpca_hyperparameter_tuning(fd, threshold=0.9):
    """
    Determine optimal number of components using the elbow method, then run FPCA with it
    Args:
        fd (FDataGrid): the original signal
        threshold (float): the threshold for the cumulative variance ratio
    Returns:
        Optimal number of components
    """
    # Centering the data to have the same starting point - 0
    # data_matrix = fd.data_matrix.squeeze()
    # data_matrix = data_matrix - data_matrix[:, 0]
    # fd = FDataGrid(data_matrix=data_matrix)

    # Find optimal number of components with elbow method
    max_components = 20
    fpca_ = FPCA(n_components=max_components)
    fpca_.fit(fd)
    var_ratio = fpca_.explained_variance_ratio_
    cumsum_var_ratio = np.cumsum(var_ratio)
    n_components = np.argmin(cumsum_var_ratio >= threshold) + 1
    return n_components

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
    # data_matrix = fd.data_matrix
    # mean = np.mean(data_matrix, axis=0)
    # data_matrix = data_matrix - mean
    # fd = fd.copy(data_matrix=data_matrix)

    # FPCA with optimal number of components
    fpca_ = FPCA(n_components=n_components).fit(fd)
    scores = fpca_.transform(fd)
    var_ratio = fpca_.explained_variance_ratio_
    mean = fpca_.mean_
    components = fpca_.components_
    return mean, components, scores, var_ratio, fpca_

#--- Inverse FPCA ---- #
def inverse_fpca(scores, components, mean):
    mean_data = mean.data_matrix
    components_data = components.data_matrix
    reconstructed_data = scores @ components_data + mean_data
    reconstructed = FDataGrid(data_matrix=reconstructed_data)
    return reconstructed
    return (scores @ components + mean).transform(warping)
from math import e
import skfda
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import get_sr
from preprocess import basis_smoothing, basis_smoothing_with_lambda, elastic_registration
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

def fpca(fd, elbow_threshold=0.005):
    """
    Determine optimal number of components using the elbow method, then run FPCA with it
    Args:
        fd (FDataGrid): the original signal
        var_threshold (float): the threshold for the cumulative variance ratio
    Returns:
        mean (FDataGrid): the mean curve
        components (FDataGrid): the components
        scores (numpy.ndarray): the scores
        var_ratio (numpy.ndarray): the variance ratio
        fpca_ (FPCA): the FPCA object
        n_components: Number of eigenfunctions obtained via elbow method
    """
    # Centering the data by subtracting the mean
    data_matrix = fd.data_matrix
    mean = np.mean(data_matrix, axis=0)
    data_matrix = data_matrix - mean
    fd = fd.copy(data_matrix=data_matrix)

    # Find optimal number of components with cumulative variance ratio and elbow method
    max_components = 10
    fpca_ = FPCA(n_components=max_components)
    scores = fpca_.fit_transform(fd)
    var_ratio = fpca_.explained_variance_ratio_
    # elbow method: find the first index where the difference in variance ratio is less than the threshold
    n_components = np.argmin(np.diff(var_ratio) < elbow_threshold) + 1

    # FPCA with optimal number of components
    fpca_ = FPCA(n_components=n_components)
    scores = fpca_.fit_transform(fd)
    var_ratio = fpca_.explained_variance_ratio_
    mean = fpca_.mean_
    components = fpca_.components_
    return mean, components, scores, var_ratio, n_components

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
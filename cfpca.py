# Source: https://github.com/ezhang1218/CFPCA/tree/main

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy 
from scipy.integrate import simps
import skfda
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import (
    BSplineBasis,
    FourierBasis,
    MonomialBasis,
)
from scipy.interpolate import interp1d




# In[2]:


def check_uniformity(data, interval):
    """ Check that all observations within the data have the same length. """
    expected_length = len(interval)
    for observation in data:
        if len(observation) != expected_length:
            raise ValueError("All observations must have the same length.")


# In[3]:


def CFPCA(foreground, background, alpha, interval, centered, aligned):
    """
    Perform Contrastive Functional Principal Component Analysis (CFPCA).

    Parameters:
    - foreground (array-like): A list or array of observations representing the foreground.
    - background (array-like): A list or array of observations representing the background.
    - alpha (float): hyperparameter that controls how much of background to subtract.
    - interval (array): The interval over which the observations are observed.
    - centered (bool): If True, center the data by subtracting the mean; if False, use data as is.
    - aligned (bool): If True, assume data is already aligned; if False, perform interpolation to align data.

    Returns:
    - array: estimated eigenfunctions (to retrieve the jth eigenfunction, do result[:,j])
    - array: estimated eigenvalues
    """
    
    # Check that all observations in both foreground and background are of the same length
    check_uniformity(foreground, interval)
    check_uniformity(background, interval)

    X = np.array(foreground)
    Y = np.array(background)

    if not aligned:
        X = interpolate_data(X)
        Y = interpolate_data(Y)
    
    if not centered:
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)
    
    w = (interval[-1] - interval[0]) / len(interval)

    Vx = (1 / (len(X) - 1)) * np.dot(X.T, X)
    Vy = (1 / (len(Y) - 1)) * np.dot(Y.T, Y)

    # Perform the eigen decomposition on the contrastive covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(w * (Vx - alpha * Vy))

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    return ((w**(-1 / 2)) * sorted_eigenvectors, (w**(-1))*sorted_eigenvalues)


# In[4]:


def interpolate_data(data):
    # First check that all observations have the same length
    n_observations = len(data)
    expected_length = len(data[0])
    for observation in data:
        if len(observation) != expected_length:
            raise ValueError("All observations must have the same length.")
    
    data = np.array(data)
    n_time_points = data.shape[1]  
    interpolated_data = np.zeros_like(data)

    for idx in range(n_observations):
        observation = data[idx]
        valid_times = np.where(~np.isnan(observation))[0]
        valid_values = observation[valid_times]

        if len(valid_times) > 1:
            interp_func = interp1d(valid_times, valid_values, kind='cubic', fill_value='extrapolate')
            interpolated_data[idx] = interp_func(np.arange(n_time_points))
        else:
            interpolated_data[idx].fill(valid_values[0] if len(valid_values) > 0 else np.nan)

    return interpolated_data


# In[5]:


def CFPCA_2(foreground, background, alpha, interval, aligned, num_bases):
    """
    Alternative method for CFPCA.

    Parameters:
    - foreground (array-like): A list or array of observations representing the foreground.
    - background (array-like): A list or array of observations representing the background.
    - alpha (float): hyperparameter that controls how much of background to subtract.
    - interval (array): The interval over which the observations are observed.
    - aligned (bool): If True, assume data is already aligned; if False, perform interpolation to align data.

    Returns:
    - array: estimated eigenfunctions (to retrieve the jth eigenfunction, do result[j])
    - array: estimated eigenvalues

    """
    
    check_uniformity(foreground, interval)
    check_uniformity(background, interval)

    X = np.array(foreground)
    Y = np.array(background)

    if not aligned:
        X = interpolate_data(X)
        Y = interpolate_data(Y)
        
    fd_X = skfda.FDataGrid(foreground, interval)
    fd_Y = skfda.FDataGrid(background, interval)
    basis = skfda.representation.basis.BSplineBasis(n_basis=num_bases)
    X_basis = fd_X.to_basis(basis)
    Y_basis = fd_Y.to_basis(basis)

    X_fd_data = fd_X.data_matrix.reshape(fd_X.data_matrix.shape[:-1])
    Y_fd_data = fd_Y.data_matrix.reshape(fd_Y.data_matrix.shape[:-1])

    identity = np.eye(len(fd_X.grid_points[0]))
    
    weights = scipy.integrate.simpson(identity, fd_X.grid_points[0])

    weights_matrix = np.diag(weights)

    factorization_matrix = weights_matrix.astype(float)

    Lt = np.linalg.cholesky(factorization_matrix).T

    new_data_matrix_X = X_fd_data @ weights_matrix
    new_data_matrix_X = np.linalg.solve(Lt.T, new_data_matrix_X.T).T

    new_data_matrix_Y = Y_fd_data @ weights_matrix
    new_data_matrix_Y = np.linalg.solve(Lt.T, new_data_matrix_Y.T).T

    X_centered = new_data_matrix_X - np.mean(new_data_matrix_X, axis=0)
    Y_centered = new_data_matrix_Y - np.mean(new_data_matrix_Y, axis=0)

    Vx = (1 / (len(foreground) - 1)) * np.dot(X_centered.T, X_centered)
    Vy = (1 / (len(background) - 1)) * np.dot(Y_centered.T, Y_centered)

    # Perform the eigen decomposition on the covariance matrix V
    eigenvalues, eigenvectors = np.linalg.eig(Vx - alpha * Vy)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    components = np.linalg.solve(Lt, sorted_eigenvectors[:, :]).T

    return (components,sorted_eigenvalues)

# In[ ]:

def l2_distance(true_func, estimated_func, t):
    """
    Computes the L2 distance between the true function values and the estimated function values,
    adjusted for the interval length, considering both the original and sign-flipped versions of the estimated function.
    
    true_func: A callable function that takes np.array t and returns an array of function values.
    estimated_func: A NumPy array representing the estimated function values at points t.
    t: np.array of points representing the discretized interval over which to compute the distance.
    
    Returns the minimum of the two distances, adjusted for the discretization interval length.
    """
    true_values = true_func(t)
    
    assert len(estimated_func) == len(t), "estimated_func must have the same number of points as t"
    
    # Calculate the differences
    diff_original = true_values - estimated_func
    diff_flipped = true_values + estimated_func
    
    delta_x = np.mean(np.diff(t))
    
    distance_original = np.sqrt(np.sum(diff_original**2)* delta_x)
    distance_flipped = np.sqrt(np.sum(diff_flipped**2) * delta_x)
    
    return np.real(min(distance_original, distance_flipped))

def plot_boxplot_with_overlayed_mean(ax, data, positions, line_color, label):
    bp = ax.boxplot(data, positions=positions, patch_artist=True,
                    boxprops=dict(facecolor='#D3D3D3', alpha=0.5),  # Set box color to light gray
                    medianprops=dict(color='black'))
    
    means = [np.mean(d) for d in data]    
    ax.plot(positions, means, color=line_color, label=label)
    ax.scatter(positions, means, color=line_color)

    return bp


def calculate_scores(data_lists, eigenvector):
    """
    Calculate the PC scores for each time series represented by a list in data_lists.
    
    Parameters:
    - data_lists: List of lists, each inner list is a time series for a company.
    - eigenvector: The eigenvector to use for calculating PC scores.
    - t: The time array over which to integrate.
    
    Returns:
    - A numpy array with PC scores for each time series.
    """
    data_array = np.array(data_lists)
    
    # Calculate the integral using the trapezoidal rule for each time series
    pc_scores = np.array([simps(data_array[i] * np.real(eigenvector), dx = 1) for i in range(data_array.shape[0])])
    return pc_scores

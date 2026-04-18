import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.linalg import subspace_angles
from utils import get_sr

def euclidean(fd1, fd2):
    data1 = fd1.data_matrix.squeeze()
    data2 = fd2.data_matrix.squeeze()
    return np.linalg.norm(data1 - data2)

def abs_cosine_similarity(fd1, fd2):
    """|cos(θ)| = |a·b| / (||a|| ||b||). Each row of f1, f2 is one vector."""
    f1 = fd1.data_matrix.squeeze()
    f2 = fd2.data_matrix.squeeze()
    dot = np.sum(f1 * f2, axis=1)
    norms = np.linalg.norm(f1, axis=1) * np.linalg.norm(f2, axis=1)
    return np.abs(dot) / np.where(norms > 0, norms, 1.0)  # avoid div by zero

def krzanowski_similarity(fd1, fd2, k=None):
    """
    Compute the Krzanowski subspace similarity between two sets of eigenfunctions.
    fd1, fd2: objects with .data_matrix of shape (n_components, n_samples)
    k: number of leading eigenvectors to use (if None, use min(rank))
    Returns: Krzanowski similarity score in [0, 1] (1: identical subspaces)
    """
    X = fd1.data_matrix.squeeze()
    Y = fd2.data_matrix.squeeze()

    # Each row: eigenfunction; so shape (n_eigen, n_samples). We'll treat columns as observations.
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if Y.ndim == 1:
        Y = Y[np.newaxis, :]

    r1 = X.shape[0]
    r2 = Y.shape[0]
    if k is None:
        k = min(r1, r2)
    # QR to orthonormalize leading k eigenvectors
    Q1, _ = np.linalg.qr(X[:k, :].T)  # shape (n_samples, k)
    Q2, _ = np.linalg.qr(Y[:k, :].T)  # shape (n_samples, k)

    # Compute singular values of Q1^T Q2
    M = np.dot(Q1.T, Q2)
    s = np.linalg.svd(M, compute_uv=False)
    similarity = np.sum(s ** 2)  / k# Krzanowski's definition: mean squared singular values

    return similarity

def fisher_rao(w1, w2):
    derivative1 = np.sqrt(w1.derivative(order=1).data_matrix.squueze())
    derivative2 = np.sqrt(w2.derivative(order=1).data_matrix.squeeze())
    return np.arccos(np.sum(derivative1 * derivative2, axis=0))
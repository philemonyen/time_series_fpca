import numpy as np
from sklearn.decomposition import PCA


def pca_n_components(data, threshold=0.90):
    """
    Choose the smallest number of principal components whose explained
    variance sums to at least `threshold`.
    """
    n_samples, n_features = data.shape
    max_components = min(n_samples, n_features)
    pca_full = PCA(n_components=max_components)
    pca_full.fit(data)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumsum, threshold) + 1)
    n_components = max(1, min(n_components, max_components))
    return n_components


def pca(data, threshold=0.90):
    """
    Fit PCA with a cumulative-variance cutoff.

    Picks the smallest n_components such that the sum of explained variance
    ratios is at least `threshold` (default 0.90), then returns the score
    matrix. Scores are the linear-combination weights in the principal-
    component basis: X_centered ≈ scores @ components_.

    Args:
        data (ndarray): raw time-series, shape (n_samples, n_times).
        threshold (float): cumulative explained-variance target in (0, 1].

    Returns:
        weights (ndarray): PCA scores, shape (n_samples, n_components).
        pca_model (PCA): fitted sklearn PCA. `pca_model.components_` is the
            basis; reuse it on new samples via pca_transform.
    """
    n_components = pca_n_components(data, threshold=threshold)
    pca_model = PCA(n_components=n_components)
    weights = pca_model.fit_transform(data)
    return weights, pca_model


def pca_transform(data, pca_model):
    """
    Apply a fitted PCA transformer to another matrix.

    Args:
        data (ndarray): raw time-series, shape (n_samples, n_times).
        pca_model (PCA): transformer returned by pca.

    Returns:
        weights (ndarray): PCA scores, shape (n_samples, n_components).
    """
    data = np.atleast_2d(np.asarray(data))
    n_features = pca_model.n_features_in_
    if data.shape[1] < n_features:
        pad = np.zeros((data.shape[0], n_features - data.shape[1]), dtype=data.dtype)
        data = np.hstack((data, pad))
    return pca_model.transform(data)

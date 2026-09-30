import numpy as np
from sklearn.metrics import mean_squared_error
from FDApy.preprocessing import PSplines, MFPCA
from FDApy.representation import DenseArgvals, DenseFunctionalData, DenseValues, MultivariateFunctionalData

def tune_psplines(raw_data, degree=3):
    """Smooth each recording along its beat axis with a 1D P-spline.

    ``PSplines.fit`` treats every axis of ``y`` as a spline dimension, and
    ``predict`` returns the fitted ndarray ([P-splines](https://fdapy.readthedocs.io/en/latest/modules/autosummary/FDApy.preprocessing.PSplines.html)).
    A clinical feature is a stack of curves, shape ``(n_signals, n_beats)``,
    so only the beat axis is the predictor. Missing beats stay out of the fit
    through ``sample_weights``.
    """
    curves = np.asarray(raw_data, dtype=float)
    if curves.ndim == 1:
        curves = curves[np.newaxis, :]
    n_signals, n_beats = curves.shape
    observed = np.isfinite(curves)
    y = np.where(observed, curves, 0.0)
    weights = observed.astype(float)
    x = np.linspace(0.0, 1.0, n_beats)

    n_observed = int(weights.sum(axis=1).max()) if n_signals else 0
    n_segments_grid = [
        int(n) for n in np.arange(10, 50, 10) if n + degree < max(n_observed, degree + 2)
    ]
    if not n_segments_grid:
        n_segments_grid = [max(1, n_observed - degree - 1)]

    best_error = float("inf")
    best_smoothed = None
    for n_segments in n_segments_grid:
        smoothed = np.empty_like(y)
        for i in range(n_signals):
            if weights[i].sum() <= degree:
                smoothed[i] = y[i]
                continue
            smoother = PSplines(n_segments=n_segments, degree=degree)
            smoother.fit(y[i], x, sample_weights=weights[i])
            smoothed[i] = np.asarray(smoother.predict(), dtype=float)

        if observed.any():
            mse = mean_squared_error(curves[observed], smoothed[observed])
        else:
            mse = float("inf")
        if mse < best_error:
            best_error = mse
            best_smoothed = smoothed

    return best_smoothed


def match_beat_length(features, n_beats):
    """Pad or trim the beat axis onto the real-data length.

    ``get_clinical_features_from_pqrst`` pads within one call, so a flawed
    batch can have a different ``max_beats`` from the data used to fit MFPCA.
    Extra beats are dropped from the right. Missing beats are NaN and are
    ignored by ``tune_psplines``.
    """
    features = np.asarray(features, dtype=float)
    width = features.shape[-1]
    if width == n_beats:
        return features
    if width > n_beats:
        return features[..., :n_beats]
    pad_width = [(0, 0)] * (features.ndim - 1) + [(0, n_beats - width)]
    return np.pad(features, pad_width, constant_values=np.nan)


def as_multivariate_functional(features):
    """Stack smoothed ``(n_signals, n_beats)`` features into one multivariate object."""
    functional = []
    for values in features:
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        argvals = DenseArgvals({"input_dim_0": np.linspace(0.0, 1.0, values.shape[1])})
        functional.append(DenseFunctionalData(argvals, DenseValues(values)))
    return MultivariateFunctionalData(functional)

def tune_n_components(multivariate_data, method="inner-product"):
    n_components = 10
    best_scores = None
    best_mfpca = None
    min_error = float('inf')

    mfpca = MFPCA(n_components=n_components, method=method)
    mfpca.fit(multivariate_data)
    scores = mfpca.transform(method="InnPro")
    error = mean_squared_error(scores, scores)
    if error < min_error:
        min_error = error
        best_scores = scores
        best_mfpca = mfpca

    return best_scores, best_mfpca

   
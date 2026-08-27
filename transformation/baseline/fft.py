import numpy as np


class FFTBasis:
    """
    Fourier basis for the first k rFFT modes.

    Attributes:
        k (int): number of retained frequency bins.
        n_times (int): series length the basis was built for.
        basis (ndarray): linear-combination basis, shape (2 * k, n_times).
            Rows are the cosine then sine (negative) modes, matching numpy.fft.rfft
            so that coefficients = data @ basis.T.
    """

    def __init__(self, k, n_times, basis):
        self.k = int(k)
        self.n_times = int(n_times)
        self.basis = basis


def _fourier_basis(n_times, k):
    t = np.arange(n_times)
    n = np.arange(k)
    angles = 2.0 * np.pi * n[:, None] * t[None, :] / n_times
    return np.vstack([np.cos(angles), -np.sin(angles)])


def fft(data, k):
    """
    Fit FFT feature extraction: project each series onto the first k Fourier modes.

    Harmonic n corresponds to the pair a_n cos(2 pi n t / T) + b_n sin(2 pi n t / T).
    The coefficient matrix stacks the real (cosine) and imaginary (sine) rFFT
    coefficients of those k lowest-frequency bins, including DC.

    Args:
        data (ndarray): raw time-series, shape (n_samples, n_times).
        k (int): number of Fourier modes (rFFT bins) to keep.

    Returns:
        coefficients (ndarray): shape (n_samples, 2 * k). Columns are
            [Re(c_0), ..., Re(c_{k-1}), Im(c_0), ..., Im(c_{k-1})].
        fft_basis (FFTBasis): fitted Fourier basis, reusable via fft_transform.
    """
    k = int(k)
    if k < 1:
        raise ValueError("k must be a positive integer")

    n_times = data.shape[1]
    spectrum = np.fft.rfft(data, axis=1)
    k = min(k, spectrum.shape[1])
    spectrum_k = spectrum[:, :k]
    coefficients = np.concatenate([spectrum_k.real, spectrum_k.imag], axis=1)
    fft_basis = FFTBasis(k=k, n_times=n_times, basis=_fourier_basis(n_times, k))
    return coefficients, fft_basis


def fft_transform(data, fft_basis):
    """
    Apply a fitted FFT basis to another matrix of the same series length.

    Args:
        data (ndarray): raw time-series, shape (n_samples, n_times).
        fft_basis (FFTBasis): transformer returned by fft.

    Returns:
        coefficients (ndarray): shape (n_samples, 2 * k).
    """
    spectrum_k = np.fft.rfft(data, axis=1)[:, :fft_basis.k]
    return np.concatenate([spectrum_k.real, spectrum_k.imag], axis=1)

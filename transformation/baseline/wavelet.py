import numpy as np


class WaveletBasis:
    """
    Band-averaged Morlet CWT filters used as a linear wavelet transformer.

    Attributes:
        frequency_bands (list): resolved (f_low, f_high) pairs in Hz.
        frequencies (list[ndarray]): CWT scales sampled inside each band.
        sampling_rate (float): sampling frequency in Hz.
        n_times (int): series length the filters were built for.
        wavelet_hat (ndarray): frequency-domain filters, shape (n_bands, n_times).
        basis (ndarray): time-domain wavelets, shape (n_bands, n_times), complex.
            Circular convolution with each row implements that band's CWT.
    """

    def __init__(self, frequency_bands, frequencies, sampling_rate, n_times, wavelet_hat, basis):
        self.frequency_bands = frequency_bands
        self.frequencies = frequencies
        self.sampling_rate = float(sampling_rate)
        self.n_times = int(n_times)
        self.wavelet_hat = wavelet_hat
        self.basis = basis


def _normalize_bands(frequency_bands):
    bands = []
    for band in frequency_bands:
        if np.isscalar(band):
            freq = float(band)
            bands.append((freq, freq))
        else:
            f_lo, f_hi = float(band[0]), float(band[1])
            if f_hi < f_lo:
                f_lo, f_hi = f_hi, f_lo
            bands.append((f_lo, f_hi))
    if not bands:
        raise ValueError("frequency_bands must contain at least one band or center frequency")
    return bands


def _morlet_hat(n_times, sampling_rate, frequencies, omega0=6.0):
    """Frequency-domain Morlet filters, shape (n_frequencies, n_times)."""
    dt = 1.0 / sampling_rate
    omega = 2.0 * np.pi * np.fft.fftfreq(n_times, d=dt)
    hats = np.empty((len(frequencies), n_times), dtype=np.complex128)
    for i, freq in enumerate(frequencies):
        scale = omega0 / (2.0 * np.pi * max(freq, 1e-12))
        wavelet_hat = (
            np.pi ** (-0.25)
            * np.sqrt(max(scale, 1e-12))
            * np.exp(-0.5 * (scale * omega - omega0) ** 2)
        )
        wavelet_hat[omega < 0] = 0.0
        hats[i] = wavelet_hat
    return hats


def _band_frequencies(bands, n_times, sampling_rate, n_scales):
    nyquist = 0.5 * sampling_rate
    min_freq = sampling_rate / n_times
    resolved_bands = []
    frequencies = []
    for f_lo, f_hi in bands:
        f_lo = min(max(f_lo, min_freq), nyquist * 0.99)
        f_hi = min(max(f_hi, f_lo), nyquist * 0.99)
        if np.isclose(f_lo, f_hi):
            freqs = np.array([f_lo], dtype=float)
        else:
            freqs = np.geomspace(f_lo, f_hi, num=max(int(n_scales), 1))
        resolved_bands.append((float(f_lo), float(f_hi)))
        frequencies.append(freqs)
    return resolved_bands, frequencies


def _apply_wavelet_hat(data, wavelet_hat):
    x_hat = np.fft.fft(data, axis=1)
    blocks = []
    for psi in wavelet_hat:
        cwt = np.fft.ifft(x_hat * psi, axis=1)
        blocks.append(np.concatenate([cwt.real, cwt.imag], axis=1))
    return np.concatenate(blocks, axis=1)


def wavelet(data, frequency_bands, sampling_rate=100.0, n_scales=8):
    """
    Fit wavelet feature extraction: CWT coefficients pooled over each frequency band.

    Each band (f_low, f_high) is covered by log-spaced Morlet scales. The complex
    CWT is averaged across those scales, then real (cosine-like) and imaginary
    (sine-like) parts are concatenated.

    Args:
        data (ndarray): raw time-series, shape (n_samples, n_times).
        frequency_bands (list): center frequencies in Hz, or (f_low, f_high) pairs.
        sampling_rate (float): sampling frequency in Hz (PTB-XL default 100).
        n_scales (int): number of CWT scales sampled inside a non-degenerate band.

    Returns:
        weights (ndarray): shape (n_samples, n_bands * 2 * n_times).
            For band b the block is [Re(cwt_b), Im(cwt_b)].
        wavelet_basis (WaveletBasis): fitted CWT filters, reusable via wavelet_transform.
    """
    n_times = data.shape[1]
    bands = _normalize_bands(frequency_bands)
    resolved_bands, frequencies = _band_frequencies(
        bands, n_times, sampling_rate, n_scales
    )
    wavelet_hat = np.stack(
        [
            np.mean(_morlet_hat(n_times, sampling_rate, freqs), axis=0)
            for freqs in frequencies
        ],
        axis=0,
    )
    basis = np.fft.ifft(wavelet_hat, axis=1)
    wavelet_basis = WaveletBasis(
        frequency_bands=resolved_bands,
        frequencies=frequencies,
        sampling_rate=sampling_rate,
        n_times=n_times,
        wavelet_hat=wavelet_hat,
        basis=basis,
    )
    weights = _apply_wavelet_hat(data, wavelet_hat)
    return weights, wavelet_basis


def wavelet_transform(data, wavelet_basis):
    """
    Apply a fitted wavelet basis to another matrix of the same series length.

    Args:
        data (ndarray): raw time-series, shape (n_samples, n_times).
        wavelet_basis (WaveletBasis): transformer returned by wavelet.

    Returns:
        weights (ndarray): shape (n_samples, n_bands * 2 * n_times).
    """
    return _apply_wavelet_hat(data, wavelet_basis.wavelet_hat)

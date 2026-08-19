# Controlled Flaw Modelling Implementation inspired by STEB https://arxiv.org/abs/2505.21160
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

def oversmoothing(data, landmarks, window_size=4):
    """
    Simulates a generative model that fails to capture high-frequency variance
    by over-smoothing the synthetic functional data.

    `window_size` is in samples on raw 10 s x 100 Hz strips (1 sample = 10 ms).
    QRS is ~8-12 samples, so 4 pts round notches and 20 pts flatten ST/T.
    """    
    # Apply a moving average across the time dimension (axis=1)
    # The 'nearest' mode handles edge effects cleanly
    smoothed_data = uniform_filter1d(data, size=window_size, axis=1, mode='nearest')
    
    return smoothed_data, landmarks

def full_memorization(real_data, substitute_data, real_landmarks, substitute_landmarks, fraction=0.02):
    """
    Simulates catastrophic privacy failure by injecting exact real samples 
    into the synthetic dataset.
    """
    n_syn = substitute_data.shape[0]
    n_real = real_data.shape[0]
    num_replace = int(n_syn * fraction)

    synthetic_data = substitute_data.copy()
    synthetic_landmarks = substitute_landmarks.copy()
    
    # Select random indices without replacement
    replace_idx = np.random.choice(n_syn, num_replace, replace=False)
    real_idx = np.random.choice(n_real, num_replace, replace=False)
    
    # Overwrite the synthetic curves with the real curves
    synthetic_data[replace_idx] = real_data[real_idx]
    synthetic_landmarks[replace_idx] = real_landmarks[real_idx]
    return synthetic_data, synthetic_landmarks

def gaussian_noise(real_data, landmarks, noise_multiplier=0.05):
    """
    Adds Gaussian noise scaled to the active (non-padded) ECG amplitude.

    `noise_multiplier` is a multiple of that std: ~0.05 is subtle EMG-scale
    artifact; ~1.0 is poor-quality but QRS-visible noise. Isoelectric padding
    is excluded so zeros do not shrink the noise scale.
    """
    active = np.abs(real_data) > 1e-6
    data_std = np.std(real_data[active]) if np.any(active) else np.std(real_data)
    noise = np.random.normal(loc=0.0, scale=data_std * noise_multiplier, size=real_data.shape)

    synthetic_data = real_data + noise

    return synthetic_data, landmarks

def mode_collapse(real_data, real_landmarks, num_modes=5, target_size=500, spike_ratio=0.10):
    """
    Creates a dataset exhibiting mode collapse by forcing regional density spikes 
    while maintaining a fixed total population size.
    """
    total_real_samples = real_data.shape[0]
    
    # Calculate partition sizes
    spike_size = int(target_size * spike_ratio) # e.g., 500 * 0.10 = 50 copies per mode
    total_spike_samples = num_modes * spike_size
    remaining_samples = target_size - total_spike_samples
    
    if total_spike_samples > target_size:
        raise ValueError("Total spike samples exceed target dataset size. Reduce num_modes or spike_ratio.")
    if remaining_samples > (total_real_samples - num_modes):
        raise ValueError("Not enough real data to fill the remaining population without replacement.")

    # 1. Select the modes (templates for the spikes)
    mode_indices = np.random.choice(total_real_samples, size=num_modes, replace=False)
    templates = real_data[mode_indices]
    templates_landmarks = real_landmarks[mode_indices]
    
    # 2. Duplicate them to create the regional spikes
    collapsed_data = np.repeat(templates, spike_size, axis=0)
    collapsed_landmarks = np.repeat(templates_landmarks, spike_size, axis=0)

    # Add microscopic noise to the collapsed samples so they aren't perfectly identical
    micro_noise = np.random.normal(0, np.std(real_data) * 0.005, size=collapsed_data.shape)
    collapsed_data += micro_noise
    
    # 3. Fill the rest of the population with a random subset of the remaining real data
    available_indices = np.setdiff1d(np.arange(total_real_samples), mode_indices)
    rest_indices = np.random.choice(available_indices, size=remaining_samples, replace=False)
    
    rest_data = real_data[rest_indices]
    rest_landmarks = real_landmarks[rest_indices]
    
    # 4. Combine the collapsed data and the remaining diverse data
    synthetic_data = np.vstack((collapsed_data, rest_data))
    synthetic_landmarks = np.vstack((collapsed_landmarks, rest_landmarks))

    
    # 5. Shuffle the dataset to randomly distribute the spikes throughout the arrays
    shuffle_idx = np.random.permutation(target_size)
    synthetic_data = synthetic_data[shuffle_idx]
    synthetic_landmarks = synthetic_landmarks[shuffle_idx]

    return synthetic_data, synthetic_landmarks


def _copy_landmark_table(landmarks):
    if isinstance(landmarks, np.ndarray) and landmarks.dtype != object:
        return landmarks.copy()
    return np.array([np.array(m, copy=True) for m in landmarks], dtype=object)


def _replace_landmarks_in_interval(syn_marks, real_marks, start_idx, end_idx):
    """
    Replace synthetic landmarks whose sample index falls in [start_idx, end_idx)
    with the real landmarks that occupy the same index range.

    Landmark values are sample indices into the data vector (e.g. 254 -> data[254]).
    Supports 1D index arrays and (n_beats, 3) P/R/T rows (membership by R-peak).
    """
    syn_marks = np.array(syn_marks, copy=True)
    real_marks = np.asarray(real_marks)

    if syn_marks.ndim == 2 and syn_marks.shape[-1] == 3:
        syn_in = (syn_marks[:, 1] >= start_idx) & (syn_marks[:, 1] < end_idx)
        real_in = (real_marks[:, 1] >= start_idx) & (real_marks[:, 1] < end_idx)
        syn_pos = np.where(syn_in)[0]
        real_pos = np.where(real_in)[0]
        if len(syn_pos) == 0 or len(real_pos) == 0:
            return syn_marks
        for i, sp in enumerate(syn_pos):
            syn_marks[sp] = real_marks[real_pos[min(i, len(real_pos) - 1)]]
        return syn_marks

    syn_flat = syn_marks.reshape(-1)
    real_flat = real_marks.reshape(-1)
    syn_in = (syn_flat >= start_idx) & (syn_flat < end_idx)
    real_in = (real_flat >= start_idx) & (real_flat < end_idx)
    syn_pos = np.where(syn_in)[0]
    real_pos = np.where(real_in)[0]
    if len(syn_pos) == 0 or len(real_pos) == 0:
        return syn_marks
    for i, sp in enumerate(syn_pos):
        syn_flat[sp] = real_flat[real_pos[min(i, len(real_pos) - 1)]]
    return syn_flat.reshape(syn_marks.shape)


def segment_leaking(real_data, substitute_data, real_landmarks, substitute_landmarks, fraction=0.05, n_beats=1, samples_per_beat=80):
    """
    Simulates partial memorization by splicing real cardiac cycles into synthetic
    traces and replacing the landmarks that live in the same sample-index range.

    Landmark values are indices into the data vector, so a leaked slice
    [start, end) takes both the waveform samples and the landmarks in that slice.

    Default `samples_per_beat=80` is one RR at 75 bpm on 100 Hz data (800 ms).
    """
    syn_data = substitute_data.copy()
    syn_landmarks = _copy_landmark_table(substitute_landmarks)

    n_syn = syn_data.shape[0]
    t_steps = syn_data.shape[1]
    n_real = real_data.shape[0]

    num_leak = int(n_syn * fraction)

    syn_idx_to_modify = np.random.choice(n_syn, num_leak, replace=False)
    real_idx_to_source = np.random.choice(n_real, num_leak, replace=False)

    beat_length = n_beats * samples_per_beat

    for s_idx, r_idx in zip(syn_idx_to_modify, real_idx_to_source):

        # 1. Isolate the active (non-padded) region for both signals
        real_trace = np.squeeze(real_data[r_idx])
        syn_trace = np.squeeze(syn_data[s_idx])
        real_active = np.abs(real_trace) > 1e-6
        syn_active = np.abs(syn_trace) > 1e-6

        real_last_idx = np.where(real_active)[0][-1] if np.any(real_active) else 0
        syn_last_idx = np.where(syn_active)[0][-1] if np.any(syn_active) else 0
        max_valid_idx = min(real_last_idx, syn_last_idx)

        if max_valid_idx < 4:
            continue

        seg_length = min(beat_length, max_valid_idx)
        if seg_length < 2:
            continue

        # 2. Pick a random starting point strictly within the active morphological region
        start_idx = np.random.randint(0, max_valid_idx - seg_length + 1)
        end_idx = start_idx + seg_length

        # 3. Splice the real cardiac-cycle segment and its landmarks
        syn_data[s_idx, start_idx:end_idx, ...] = real_data[r_idx, start_idx:end_idx, ...]
        syn_landmarks[s_idx] = _replace_landmarks_in_interval(
            syn_landmarks[s_idx], real_landmarks[r_idx], start_idx, end_idx
        )

    return syn_data, syn_landmarks

def _power_warp_indices(marks, alpha, t_steps):
    denom = max(t_steps - 1, 1)
    u = np.clip(np.asarray(marks, dtype=float) / denom, 0.0, 1.0)
    return np.clip(np.rint((u ** (1.0 / alpha)) * denom), 0, t_steps - 1).astype(int)


def time_distortion(data, landmarks, alpha=1.03):
    """
    Simulates global time distortion using a power-law warp on raw traces.

    `data` is [N, T] (10 s x 100 Hz => T=1000). Landmarks are sample indices
    into that vector. `alpha` should stay very near 1: even 1.08 shifts the
    midpoint by ~270 ms. The scale grid uses 0.94-1.06 (~110-200 ms).
    """
    n_samples, t_steps = data.shape
    grid = np.linspace(0.0, 1.0, t_steps)
    gamma_t = np.clip(grid ** alpha, 0.0, 1.0)

    distorted_data = np.empty_like(data)
    for i in range(n_samples):
        distorted_data[i] = np.interp(gamma_t, grid, data[i])

    distorted_landmarks = _copy_landmark_table(landmarks)
    for i in range(n_samples):
        marks = np.asarray(landmarks[i])
        distorted_landmarks[i] = _power_warp_indices(marks, alpha, t_steps).reshape(marks.shape)

    return distorted_data, distorted_landmarks


def phase_shift(data, landmarks, shift_fraction=0.05):
    """
    Simulates a systematic phase shift of the internal beats on raw traces.

    `data` is [N, T]. Landmarks are sample indices. `shift_fraction` is a
    fraction of the local RR interval: 0.05 ≈ 40 ms and 0.15 ≈ 120 ms at 75 bpm.
    Endpoints 0 and T-1 stay fixed; P/R/T are mapped through the same warp.
    """
    n_samples, t_steps = data.shape
    grid = np.arange(t_steps, dtype=float)

    distorted_data = np.empty_like(data)
    distorted_landmarks = _copy_landmark_table(landmarks)

    for i in range(n_samples):
        marks = np.asarray(landmarks[i], dtype=float)
        r_peaks = marks[:, 1] if (marks.ndim == 2 and marks.shape[-1] == 3) else marks.reshape(-1)
        r_peaks = np.clip(np.sort(r_peaks), 0.0, float(t_steps - 1))

        orig_marks = np.unique(np.concatenate(([0.0], r_peaks, [float(t_steps - 1)])))
        new_marks = orig_marks.copy()
        rr_intervals = np.diff(orig_marks)
        mean_rr = np.mean(rr_intervals) if len(rr_intervals) else 0.0
        base_shift = shift_fraction * mean_rr

        for j in range(1, len(orig_marks) - 1):
            shift = base_shift
            if new_marks[j] + shift >= orig_marks[j + 1]:
                shift = (orig_marks[j + 1] - orig_marks[j]) * 0.1
            elif new_marks[j] + shift <= new_marks[j - 1]:
                shift = (orig_marks[j] - new_marks[j - 1]) * 0.1
            new_marks[j] += shift

        gamma = interp1d(new_marks, orig_marks, kind="linear", bounds_error=False, fill_value="extrapolate")
        inv_gamma = interp1d(orig_marks, new_marks, kind="linear", bounds_error=False, fill_value="extrapolate")

        query = np.clip(gamma(grid), 0.0, float(t_steps - 1))
        distorted_data[i] = np.interp(query, grid, data[i])

        warped = np.clip(np.rint(inv_gamma(marks.reshape(-1))), 0, t_steps - 1).astype(int)
        distorted_landmarks[i] = warped.reshape(np.asarray(landmarks[i]).shape)

    return distorted_data, distorted_landmarks
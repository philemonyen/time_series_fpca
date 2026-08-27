# Controlled Flaw Modelling Implementation inspired by STEB https://arxiv.org/abs/2505.21160
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

### Morphological Flawed Scenario Engineering ###
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

### Distributional Flawed Scenario Engineering ###
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

def baseline_drift(data, landmarks, duration_sec=10.0, drift_freq=0.2, amplitude_fraction=0.5):
    """
    Simulates a low-frequency wandering baseline.
    
    `drift_freq` is the frequency of the drift in Hz (typically 0.1 to 0.5).
    `amplitude_fraction` scales the drift relative to the signal's standard deviation.
    Landmarks are returned unmodified.
    """
    n_samples, t_steps = data.shape
    t = np.linspace(0, duration_sec, t_steps)
    
    distorted_data = np.empty_like(data)
    
    for i in range(n_samples):
        sig_std = np.std(data[i])
        drift_amplitude = sig_std * amplitude_fraction
        
        # Randomize phase so not all records drift in the exact same direction
        phase = np.random.uniform(0, 2 * np.pi)
        
        drift = drift_amplitude * np.sin(2 * np.pi * drift_freq * t + phase)
        distorted_data[i] = data[i] + drift

    # Time domain is untouched, copy landmarks directly
    distorted_landmarks = _copy_landmark_table(landmarks)
    
    return distorted_data, distorted_landmarks

def spurious_transient(data, landmarks, amplitude_fraction=2.0, num_spikes=1):
    """
    Simulates spurious transients (hallucinated sharp spikes or notches).
    
    `amplitude_fraction` scales the height of the injected spike relative to the 
    signal's standard deviation.
    `num_spikes` determines how many transients to randomly inject per 10s record.
    """
    n_samples, t_steps = data.shape
    distorted_data = np.empty_like(data)
    distorted_landmarks = _copy_landmark_table(landmarks)
    
    for i in range(n_samples):
        sig_std = np.std(data[i])
        distorted_data[i] = data[i].copy()
        
        for _ in range(num_spikes):
            # Pick a random location (keep slightly away from the absolute edges)
            spike_center = np.random.randint(10, t_steps - 10)
            
            # Create a narrow spike (Gaussian with standard deviation of 1 to 3 samples, i.e., 10-30 ms)
            x = np.arange(t_steps)
            spike_width = np.random.uniform(1.0, 3.0) 
            spike_shape = np.exp(-0.5 * ((x - spike_center) / spike_width) ** 2)
            
            # Randomize polarity (it can be a spike pointing up or a notch pointing down)
            polarity = np.random.choice([-1, 1])
            
            # Inject the hallucination
            spike_amplitude = polarity * sig_std * amplitude_fraction
            distorted_data[i] += spike_amplitude * spike_shape
            
    return distorted_data, distorted_landmarks

# Temporal Flawed Scenario Engineering ###
def _copy_landmark_table(landmarks):
    if isinstance(landmarks, np.ndarray) and landmarks.dtype != object:
        return landmarks.copy()
    return np.array([np.array(m, copy=True) for m in landmarks], dtype=object)

def _as_beat_triplets(marks):
    """Require per-record landmarks of shape (n_beats, 3) = [P, R, T]."""
    marks = np.asarray(marks)
    if marks.ndim != 2 or marks.shape[-1] != 3:
        raise ValueError(
            "Expected per-record landmarks of shape (n_beats, 3) with columns "
            f"[P-onset, R-peak, T-offset]; got shape {marks.shape}."
        )
    return marks

def _r_peak_indices(marks, t_steps):
    """R-peak sample indices, ignoring -1 padding from get_landmarks."""
    marks = _as_beat_triplets(marks).astype(float, copy=False)
    r_peaks = marks[marks[:, 1] >= 0, 1]
    if r_peaks.size == 0:
        return r_peaks
    return np.clip(np.sort(r_peaks), 0.0, float(t_steps - 1))

def _map_valid_indices(marks, mapper, t_steps):
    """Apply mapper to valid (>=0) landmark indices; keep -1 padding as-is.

    align_ecg drops unused beat slots with `R < 0`. Clipping or interpolating
    those -1 sentinels maps them to 0, which then becomes a fake last beat
    and yields an empty trim (`start_idx > end_idx`).
    """
    original = _as_beat_triplets(marks)
    values = original.astype(float, copy=False).reshape(-1)
    warped = original.copy().reshape(-1)
    valid = values >= 0
    if np.any(valid):
        mapped = np.clip(np.rint(mapper(values[valid])), 0, t_steps - 1)
        warped[valid] = mapped.astype(warped.dtype, copy=False)
    return warped.reshape(original.shape)

def _strictly_increasing(marks, min_step=1e-6):
    out = np.asarray(marks, dtype=float).copy()
    for j in range(1, len(out)):
        if out[j] <= out[j - 1]:
            out[j] = out[j - 1] + min_step
    return out

def _power_warp_indices(marks, alpha, t_steps):
    denom = max(t_steps - 1, 1)

    def _warp(values):
        u = np.clip(values / denom, 0.0, 1.0)
        return (u ** (1.0 / alpha)) * denom

    return _map_valid_indices(marks, _warp, t_steps)

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
        distorted_landmarks[i] = _power_warp_indices(landmarks[i], alpha, t_steps)

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
        r_peaks = _r_peak_indices(marks, t_steps)

        orig_marks = np.unique(np.concatenate(([0.0], r_peaks, [float(t_steps - 1)])))
        new_marks = orig_marks.copy()
        
        rr_intervals = np.diff(orig_marks)
        mean_rr = np.mean(rr_intervals) if len(rr_intervals) else 0.0
        base_shift = shift_fraction * mean_rr

        # Strictly enforce monotonicity
        for j in range(1, len(orig_marks) - 1):
            proposed_pos = orig_marks[j] + base_shift
            
            # Create a safe boundary (10% padding from adjacent peaks)
            left_margin = (orig_marks[j] - orig_marks[j-1]) * 0.1
            right_margin = (orig_marks[j+1] - orig_marks[j]) * 0.1
            
            min_allowed = new_marks[j-1] + left_margin
            max_allowed = orig_marks[j+1] - right_margin
            
            # Fallback in case peaks are extremely compressed
            if min_allowed >= max_allowed:
                new_marks[j] = (new_marks[j-1] + orig_marks[j+1]) / 2.0
            else:
                new_marks[j] = np.clip(proposed_pos, min_allowed, max_allowed)

        orig_marks = _strictly_increasing(orig_marks)
        new_marks = _strictly_increasing(new_marks)

        # Interpolation will now strictly preserve topology
        gamma = interp1d(new_marks, orig_marks, kind="linear", bounds_error=False, fill_value="extrapolate")
        inv_gamma = interp1d(orig_marks, new_marks, kind="linear", bounds_error=False, fill_value="extrapolate")

        query = np.clip(gamma(grid), 0.0, float(t_steps - 1))
        distorted_data[i] = np.interp(query, grid, data[i])
        distorted_landmarks[i] = _map_valid_indices(landmarks[i], inv_gamma, t_steps)

    return distorted_data, distorted_landmarks

def phase_jitter(data, landmarks, jitter_fraction=0.05):
    """
    Simulates random phase jitter (desynchronization) of internal beats.

    `data` is [N, T]. Landmarks are sample indices. `jitter_fraction` defines
    the standard deviation of the random shift as a fraction of the RR interval.
    Unlike phase_shift which shifts systematically, this randomizes each beat.
    """
    n_samples, t_steps = data.shape
    grid = np.arange(t_steps, dtype=float)

    distorted_data = np.empty_like(data)
    distorted_landmarks = _copy_landmark_table(landmarks)

    for i in range(n_samples):
        marks = np.asarray(landmarks[i], dtype=float)
        r_peaks = _r_peak_indices(marks, t_steps)

        orig_marks = np.unique(np.concatenate(([0.0], r_peaks, [float(t_steps - 1)])))
        new_marks = orig_marks.copy()
        rr_intervals = np.diff(orig_marks)
        mean_rr = np.mean(rr_intervals) if len(rr_intervals) else 0.0
        
        # Apply random jitter to each internal peak
        for j in range(1, len(orig_marks) - 1):
            # Draw random shift from normal distribution
            shift = np.random.normal(loc=0.0, scale=jitter_fraction * mean_rr)
            
            # Constrain shift to prevent peak overlap (maintain causality)
            max_forward = (orig_marks[j + 1] - orig_marks[j]) * 0.4
            max_backward = (orig_marks[j] - new_marks[j - 1]) * 0.4
            shift = np.clip(shift, -max_backward, max_forward)
            
            new_marks[j] += shift

        orig_marks = _strictly_increasing(orig_marks)
        new_marks = _strictly_increasing(new_marks)

        gamma = interp1d(new_marks, orig_marks, kind="linear", bounds_error=False, fill_value="extrapolate")
        inv_gamma = interp1d(orig_marks, new_marks, kind="linear", bounds_error=False, fill_value="extrapolate")

        query = np.clip(gamma(grid), 0.0, float(t_steps - 1))
        distorted_data[i] = np.interp(query, grid, data[i])
        distorted_landmarks[i] = _map_valid_indices(landmarks[i], inv_gamma, t_steps)

    return distorted_data, distorted_landmarks

def loss_of_autocorrelation(data, landmarks, shuffle_ratio=0.5):
    """
    Destroys long-term temporal correlation by shuffling a proportion of heartbeats.
    
    `shuffle_ratio`: Float (0.0 to 1.0). Determines what percentage of the internal 
                     segments are selected for random shuffling. 0.0 means no flaw, 
                     1.0 means complete randomization of the sequence.
    """
    n_samples, t_steps = data.shape
    distorted_data = np.zeros_like(data)
    distorted_landmarks = _copy_landmark_table(landmarks)
    
    for i in range(n_samples):
        marks = np.asarray(landmarks[i], dtype=float)
        r_peaks = _r_peak_indices(marks, t_steps).astype(int)
        
        # If not enough peaks to shuffle, return original
        if len(r_peaks) < 2 or shuffle_ratio <= 0.0:
            distorted_data[i] = data[i]
            continue
            
        # Define cut points at the midpoint of RR intervals
        cut_points = [0]
        for j in range(len(r_peaks) - 1):
            midpoint = (r_peaks[j] + r_peaks[j+1]) // 2
            cut_points.append(midpoint)
        cut_points.append(t_steps)
        
        # Extract all segments
        segments = [data[i, cut_points[j]:cut_points[j+1]] for j in range(len(cut_points) - 1)]
        num_segments = len(segments)
        
        # Determine how many segments to shuffle based on the ratio
        num_to_shuffle = int(np.round(num_segments * shuffle_ratio))
        
        if num_to_shuffle > 1:
            # Randomly pick the indices of the segments we want to shuffle
            indices_to_shuffle = np.random.choice(num_segments, num_to_shuffle, replace=False)
            
            # Extract those specific segments
            segments_to_shuffle = [segments[idx] for idx in indices_to_shuffle]
            
            # Shuffle them
            np.random.shuffle(segments_to_shuffle)
            
            # Put them back into the selected indices
            for idx, shuffled_segment in zip(indices_to_shuffle, segments_to_shuffle):
                segments[idx] = shuffled_segment
                
        # Reassemble the sequence
        reassembled = np.concatenate(segments)
        
        # Handle tiny length mismatches due to rounding or segment swapping
        if len(reassembled) > t_steps:
            distorted_data[i] = reassembled[:t_steps]
        elif len(reassembled) < t_steps:
            distorted_data[i, :len(reassembled)] = reassembled
            distorted_data[i, len(reassembled):] = reassembled[-1]
        else:
            distorted_data[i] = reassembled

    return distorted_data, distorted_landmarks

### Privacy Flawed Scenario Engineering ###
def inject_segment_leak(canary_record, host_record, leak_ratio, transition_width=10):
    """
    Injects a prefix of the canary into the host with a smooth cross-fade.
    
    leak_ratio: Float (0.0 to 1.0) indicating how much of the sequence to leak.
    transition_width: Number of timepoints over which to blend the splice.
    """
    n_timepoints = len(canary_record)
    leaked_length = int(n_timepoints * leak_ratio)
    
    # Create the weight vector w(t)
    weights = np.zeros(n_timepoints)
    
    # 1. Fully leaked segment
    weights[:leaked_length] = 1.0
    
    # 2. Smooth transition window (linear fade)
    fade_end = min(leaked_length + transition_width, n_timepoints)
    actual_width = fade_end - leaked_length
    
    if actual_width > 0:
        fade = np.linspace(1.0, 0.0, actual_width)
        weights[leaked_length:fade_end] = fade
        
    # 3. Apply cross-fade
    flawed_record = (weights * canary_record) + ((1.0 - weights) * host_record)
    
    return flawed_record
# Controlled Flaw Modelling Implementation inspired by STEB https://arxiv.org/abs/2505.21160
import numpy as np
from skfda import FDataGrid
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

def oversmoothing(fd, window_size = 5):
    """
    Simulates a generative model that fails to capture high-frequency variance 
    by over-smoothing the synthetic functional data.
    """
    # data_matrix shape is (n_samples, n_eval_points, n_channels)
    data = fd.data_matrix 
    
    # Apply a moving average across the time dimension (axis=1)
    # The 'nearest' mode handles edge effects cleanly
    smoothed_data = uniform_filter1d(data, size=window_size, axis=1, mode='nearest')
    
    return FDataGrid(
        data_matrix=smoothed_data, 
        grid_points=fd.grid_points,
    )

def full_memorization(fd_real, fd_substitute, fraction = 0.1):
    """
    Simulates catastrophic privacy failure by injecting exact real samples 
    into the synthetic dataset.
    """
    syn_data = fd_substitute.data_matrix.copy()
    real_data = fd_real.data_matrix
    
    n_syn = syn_data.shape[0]
    n_real = real_data.shape[0]
    num_replace = int(n_syn * fraction)
    
    # Select random indices without replacement
    replace_idx = np.random.choice(n_syn, num_replace, replace=False)
    real_idx = np.random.choice(n_real, num_replace, replace=False)
    
    # Overwrite the synthetic curves with the real curves
    syn_data[replace_idx] = real_data[real_idx]
    return FDataGrid(
        data_matrix=syn_data, 
        grid_points=fd_substitute.grid_points,
    )

def gaussian_noise(real_data_fd, noise_multiplier=1.5):
    """
    Adds extreme Gaussian noise relative to the standard deviation of the data.
    """
    real_data = real_data_fd.data_matrix.squeeze()
    data_std = np.std(real_data)
    heavy_noise = np.random.normal(loc=0.0, scale=data_std * noise_multiplier, size=real_data.shape)
    
    synthetic_ds = real_data + heavy_noise

    return FDataGrid(data_matrix=synthetic_ds, grid_points=real_data_fd.grid_points)

def mode_collapse(real_data_fd, num_modes=5, target_size=500, spike_ratio=0.10):
    """
    Creates a dataset exhibiting mode collapse by forcing regional density spikes 
    while maintaining a fixed total population size.
    """
    real_data = real_data_fd.data_matrix.squeeze()
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
    
    # 2. Duplicate them to create the regional spikes
    collapsed_data = np.repeat(templates, spike_size, axis=0)
    
    # Add microscopic noise to the collapsed samples so they aren't perfectly identical
    micro_noise = np.random.normal(0, np.std(real_data) * 0.005, size=collapsed_data.shape)
    collapsed_data += micro_noise
    
    # 3. Fill the rest of the population with a random subset of the remaining real data
    available_indices = np.setdiff1d(np.arange(total_real_samples), mode_indices)
    rest_indices = np.random.choice(available_indices, size=remaining_samples, replace=False)
    
    rest_data = real_data[rest_indices]
    
    # 4. Combine the collapsed data and the remaining diverse data
    synthetic_ds = np.vstack((collapsed_data, rest_data))
    
    # 5. Shuffle the dataset to randomly distribute the spikes throughout the arrays
    shuffle_idx = np.random.permutation(target_size)
    synthetic_ds = synthetic_ds[shuffle_idx]
    
    return FDataGrid(data_matrix=synthetic_ds, grid_points=real_data_fd.grid_points)

def segment_leaking(fd_real, fd_substitute, fraction=0.1):
    """
    Simulates partial memorization by splicing segments of real data 
    directly into the synthetic time-series, while strictly avoiding 
    the zero-padded isoelectric regions.
    """
    syn_data = fd_substitute.data_matrix.copy()
    real_data = fd_real.data_matrix
    
    n_syn, t_steps, num_dim = syn_data.shape
    n_real = real_data.shape[0]
    
    num_leak = int(n_syn * fraction)
    
    syn_idx_to_modify = np.random.choice(n_syn, num_leak, replace=False)
    real_idx_to_source = np.random.choice(n_real, num_leak, replace=False)
    
    for s_idx, r_idx in zip(syn_idx_to_modify, real_idx_to_source):
        
        # 1. Isolate the active (non-padded) region for both signals
        # Using 1e-6 to account for potential floating-point inaccuracies during generation
        real_active = np.abs(real_data[r_idx, :, 0]) > 1e-6
        syn_active = np.abs(syn_data[s_idx, :, 0]) > 1e-6
        
        # Find the last active index (where the padding begins)
        real_last_idx = np.where(real_active)[0][-1] if np.any(real_active) else 0
        syn_last_idx = np.where(syn_active)[0][-1] if np.any(syn_active) else 0
        
        # The valid region for splicing is the intersection of their active lengths
        max_valid_idx = min(real_last_idx, syn_last_idx)
        
        # Safety check: if the active region is too small to meaningfully splice, skip
        if max_valid_idx < 4:
            continue
            
        # 2. Determine segment length based on the ACTIVE steps, not total t_steps
        min_len = max(1, max_valid_idx // 4)
        max_len = max(2, max_valid_idx // 2)
        
        if min_len >= max_len:
            seg_length = min_len
        else:
            seg_length = np.random.randint(min_len, max_len)
        
        # 3. Pick a random starting point strictly within the active morphological region
        start_idx = np.random.randint(0, max_valid_idx - seg_length + 1)
        end_idx = start_idx + seg_length
        
        # 4. Splice the real segment into the synthetic curve
        syn_data[s_idx, start_idx:end_idx, :] = real_data[r_idx, start_idx:end_idx, :]
        
    return FDataGrid(
        data_matrix=syn_data, 
        grid_points=fd_substitute.grid_points,
    )

def time_distortion(fd, landmarks, alpha=1.5):
    """
    Simulates global time distortion using a power-law warp.
    """
    grid = fd.grid_points[0]
    n_samples = fd.data_matrix.shape[0]
    
    # 1. Create the non-linear warping function gamma(t)
    gamma_t = grid ** alpha
    
    # 2. Warp the functional data by evaluating it at the warped time points
    distorted_data = fd(gamma_t).squeeze()
    
    # Ensure correct shape if only one sample is passed
    if n_samples == 1:
        distorted_data = distorted_data[np.newaxis, :]
        
    # 3. Update the landmarks
    # If the original peak was at L, the new peak occurs when gamma(t) = L
    # t^alpha = L  =>  t = L^(1/alpha)
    distorted_landmarks = landmarks ** (1.0 / alpha)
    
    fd_distorted = FDataGrid(data_matrix=distorted_data, grid_points=grid)
    return fd_distorted, distorted_landmarks

def phase_shift(fd, landmarks, shift_fraction=0.03):
    """
    Simulates a systematic phase shift of the internal beats (e.g., early/delayed beats)
    without moving the fixed window boundaries at t=0 and t=1.
    """
    grid = fd.grid_points[0]
    n_samples = fd.data_matrix.shape[0]
    
    distorted_data = np.zeros_like(fd.data_matrix).squeeze()
    if n_samples == 1: distorted_data = distorted_data[np.newaxis, :]
        
    distorted_landmarks = np.copy(landmarks)
    
    for i in range(n_samples):
        orig_marks = landmarks[i]
        new_marks = np.copy(orig_marks)
        
        # 1. Shift internal landmarks 
        for j in range(1, len(orig_marks) - 1):
            shift = shift_fraction
            
            # Safety check: maintain monotonicity (do not let beats cross each other)
            if new_marks[j] + shift >= orig_marks[j+1]:
                shift = (orig_marks[j+1] - orig_marks[j]) * 0.1
            elif new_marks[j] + shift <= orig_marks[j-1]:
                shift = (orig_marks[j-1] - orig_marks[j]) * 0.1
                
            new_marks[j] += shift
            
        distorted_landmarks[i] = new_marks
        
        # 2. Create a piecewise linear warping function gamma(t)
        # This maps the new timing coordinates back to the original timing coordinates
        gamma_i = interp1d(
            new_marks, 
            orig_marks, 
            kind='linear', 
            fill_value="extrapolate"
        )(grid)
        
        # Clip to ensure no floating point errors push gamma outside [0, 1]
        gamma_i = np.clip(gamma_i, 0, 1)
        
        # 3. Interpolate the original signal at the new warped grid points
        curve_interp = interp1d(grid, fd.data_matrix[i, :, 0].squeeze(), kind='linear')
        distorted_data[i] = curve_interp(gamma_i)
        
    fd_distorted = FDataGrid(data_matrix=distorted_data, grid_points=grid)
    return fd_distorted, distorted_landmarks
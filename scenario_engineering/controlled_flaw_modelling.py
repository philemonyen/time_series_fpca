# Controlled Flaw Modelling Implementation inspired by STEB https://arxiv.org/abs/2505.21160
import numpy as np
from skfda import FDataGrid
from scipy.ndimage import uniform_filter1d

def oversmoothing(fd, landmarks, window_size = 5):
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
    ), landmarks

def full_memorization(fd_real, fd_substitute, landmarks_real, landmarks_substitute, fraction = 0.1):
    """
    Simulates catastrophic privacy failure by injecting exact real samples 
    into the synthetic dataset.
    """
    syn_data = fd_substitute.data_matrix.copy()
    syn_landmarks = landmarks_substitute.copy()
    real_data = fd_real.data_matrix
    real_landmarks = landmarks_real.copy()
    
    n_syn = syn_data.shape[0]
    n_real = real_data.shape[0]
    num_replace = int(n_syn * fraction)
    
    # Select random indices without replacement
    replace_idx = np.random.choice(n_syn, num_replace, replace=False)
    real_idx = np.random.choice(n_real, num_replace, replace=False)
    
    # Overwrite the synthetic curves with the real curves
    syn_data[replace_idx] = real_data[real_idx]
    syn_landmarks[replace_idx] = real_landmarks[real_idx]
    return FDataGrid(
        data_matrix=syn_data, 
        grid_points=fd_substitute.grid_points,
    ), syn_landmarks

def gaussian_noise(real_data_fd, landmarks, noise_multiplier=1.5):
    """
    Adds extreme Gaussian noise relative to the standard deviation of the data.
    """
    real_data = real_data_fd.data_matrix.squeeze()
    data_std = np.std(real_data)
    heavy_noise = np.random.normal(loc=0.0, scale=data_std * noise_multiplier, size=real_data.shape)
    
    synthetic_ds = real_data + heavy_noise

    return FDataGrid(data_matrix=synthetic_ds, grid_points=real_data_fd.grid_points), landmarks

def mode_collapse(real_data_fd, real_landmarks, num_modes=5, target_size=500, spike_ratio=0.10):
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
    synthetic_ds = np.vstack((collapsed_data, rest_data))
    synthetic_landmarks = np.vstack((collapsed_landmarks, rest_landmarks))
    
    # 5. Shuffle the dataset to randomly distribute the spikes throughout the arrays
    shuffle_idx = np.random.permutation(target_size)
    synthetic_ds = synthetic_ds[shuffle_idx]
    synthetic_landmarks = synthetic_landmarks[shuffle_idx]
    
    return FDataGrid(data_matrix=synthetic_ds, grid_points=real_data_fd.grid_points), synthetic_landmarks

def segment_leaking(fd_real, fd_substitute, landmarks_real, landmarks_substitute, fraction = 0.1):
    """
    Simulates partial memorization by splicing segments of real data 
    directly into the synthetic time-series.
    """
    syn_data = fd_substitute.data_matrix.copy()
    syn_landmarks = landmarks_substitute.copy()
    real_data = fd_real.data_matrix
    real_landmarks = landmarks_real.copy()
    
    n_syn, t_steps, _ = syn_data.shape
    n_real = real_data.shape[0]
    
    num_leak = int(n_syn * fraction)
    
    syn_idx_to_modify = np.random.choice(n_syn, num_leak, replace=False)
    real_idx_to_source = np.random.choice(n_real, num_leak, replace=False)
    
    for s_idx, r_idx in zip(syn_idx_to_modify, real_idx_to_source):
        # Determine a random segment length between 25% and 50% of the total steps
        min_len = max(1, t_steps // 4)
        max_len = max(2, t_steps // 2)
        seg_length = np.random.randint(min_len, max_len)
        
        # Pick a random starting point that allows the full segment to fit
        start_idx = np.random.randint(0, t_steps - seg_length + 1)
        end_idx = start_idx + seg_length
        
        # Splice the real segment into the synthetic curve
        syn_data[s_idx, start_idx:end_idx, :] = real_data[r_idx, start_idx:end_idx, :]
        syn_landmarks[s_idx, start_idx:end_idx] = real_landmarks[r_idx, start_idx:end_idx]
    return FDataGrid(
        data_matrix=syn_data, 
        grid_points=fd_substitute.grid_points,
    ), syn_landmarks
import numpy as np
from skfda import FDataGrid

def create_low_fidelity_dataset(real_data_fd, noise_multiplier=1.5):
    """
    Adds extreme Gaussian noise relative to the standard deviation of the data.
    """
    real_data = real_data_fd.data_matrix.squeeze()
    data_std = np.std(real_data)
    heavy_noise = np.random.normal(loc=0.0, scale=data_std * noise_multiplier, size=real_data.shape)
    
    synthetic_ds = real_data + heavy_noise

    return FDataGrid(data_matrix=synthetic_ds, grid_points=real_data_fd.grid_points)

def create_mode_collapse_dataset(real_data_fd, real_landmarks, num_modes=5, target_size=500, spike_ratio=0.10):
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

def create_exact_memorization_dataset(real_data_fd, real_landmarks, num_memorized=10, total_synthetic=500):
    """
    Injects 1-to-1 exact replicas of real outliers into a safe synthetic set
    to test micro-scale privacy detection.
    """
    # 1. Identify Outliers in the Real Data
    # Simple metric: distance from the mean heartbeat
    real_data = real_data_fd.data_matrix.squeeze()
    mean_heartbeat = np.mean(real_data, axis=0)
    distances_to_mean = np.linalg.norm(real_data - mean_heartbeat, axis=1)
    
    # Get the indices of the most extreme outliers (highest distance from mean)
    outlier_indices = np.argsort(distances_to_mean)[-num_memorized:]
    memorized_samples = real_data[outlier_indices]
    memorized_landmarks = real_landmarks[outlier_indices]
    
    # 2. Create a "Safe" baseline synthetic set (e.g., highly smoothed real data)
    # This simulates a generalized model that hasn't memorized anything else
    safe_samples_needed = total_synthetic - num_memorized
    random_indices = np.random.choice(real_data.shape[0], size=safe_samples_needed)
    safe_baseline = real_data[random_indices] + np.random.normal(0, np.std(real_data)*0.005, (safe_samples_needed, real_data.shape[1]))
    safe_baseline_landmarks = real_landmarks[random_indices]
    # 3. Inject the exact memorized outliers
    # We do NOT add noise to these. They are exact 1-to-1 copies.
    synthetic_ds = np.vstack([safe_baseline, memorized_samples])
    synthetic_landmarks = np.vstack([safe_baseline_landmarks, memorized_landmarks])
    # Shuffle to ensure they aren't all just sitting at the bottom of the array
    np.random.shuffle(synthetic_ds)
    
    return FDataGrid(data_matrix=synthetic_ds, grid_points=real_data_fd.grid_points), synthetic_landmarks
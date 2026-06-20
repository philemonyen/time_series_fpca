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

def create_mode_collapse_dataset(real_data_fd, real_landmarks, num_templates=5, copies_per_template=100):
    """
    Forces a massive regional density spike (macro-scale privacy failure).
    """
    # 1. Randomly select 5 "template" heartbeats
    real_data = real_data_fd.data_matrix.squeeze()
    template_indices = np.random.choice(real_data.shape[0], size=num_templates, replace=False)
    templates = real_data[template_indices]
    templates_landmarks = real_landmarks[template_indices]
    
    # 2. Duplicate them
    # np.repeat copies each row 'copies_per_template' times consecutively
    collapsed_data = np.repeat(templates, copies_per_template, axis=0)
    collapsed_landmarks = np.repeat(templates_landmarks, copies_per_template, axis=0)
    # 3. Add microscopic noise (0.5% of std) so points aren't perfectly identical
    micro_noise = np.random.normal(0, np.std(real_data) * 0.005, size=collapsed_data.shape)
    
    synthetic_ds = collapsed_data + micro_noise
    synthetic_landmarks = collapsed_landmarks
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
    safe_baseline = real_data[random_indices] + np.random.normal(0, np.std(real_data)*0.5, (safe_samples_needed, real_data.shape[1]))
    safe_baseline_landmarks = real_landmarks[random_indices]
    # 3. Inject the exact memorized outliers
    # We do NOT add noise to these. They are exact 1-to-1 copies.
    synthetic_ds = np.vstack([safe_baseline, memorized_samples])
    synthetic_landmarks = np.vstack([safe_baseline_landmarks, memorized_landmarks])
    # Shuffle to ensure they aren't all just sitting at the bottom of the array
    np.random.shuffle(synthetic_ds)
    
    return FDataGrid(data_matrix=synthetic_ds, grid_points=real_data_fd.grid_points), synthetic_landmarks
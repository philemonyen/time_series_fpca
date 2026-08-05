import pickle
from pathlib import Path
from validation.controlled_flaw_modelling import *
from preprocess.utils import get_sr, load_dataset, extract_ecg_clinical_landmarks

def oversmoothing_creation(fd, landmarks):
    dataset = {}
    window_size = [5, 10, 15, 20, 25]
    for window in window_size:
        dataset[window] = oversmoothing(fd, landmarks, window)
    return dataset

def memorization_creation(fd_real, fd_substitute, landmarks_real, landmarks_substitute):
    dataset = {}
    fraction = [0.1, 0.2, 0.3, 0.4, 0.5]
    for fraction in fraction:
        dataset[fraction] = full_memorization(fd_real, fd_substitute, landmarks_real, landmarks_substitute, fraction)
    return dataset

def gaussian_noise_creation(fd, landmarks):
    dataset = {}
    noise_multiplier = [1.5, 2.0, 2.5, 3.0, 3.5]
    for noise_multiplier in noise_multiplier:
        dataset[noise_multiplier] = gaussian_noise(fd, landmarks, noise_multiplier)
    return dataset

def mode_collapse_vary_modes_creation(fd, landmarks):
    dataset = {}
    num_modes = [1, 2, 3, 4, 5]
    for num_modes in num_modes:
        dataset[num_modes] = mode_collapse(fd, landmarks, num_modes=num_modes)
    return dataset

def mode_collapse_vary_spike_ratio_creation(fd, landmarks):
    dataset = {}
    spike_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]
    for spike_ratio in spike_ratio:
        dataset[spike_ratio] = mode_collapse(fd, landmarks, num_modes=1, spike_ratio=spike_ratio)
    return dataset

def segment_leaking_creation(fd_real, fd_substitute, landmarks_real, landmarks_substitute):
    dataset = {}
    fraction = [0.1, 0.2, 0.3, 0.4, 0.5]
    for fraction in fraction:
        dataset[fraction] = segment_leaking(fd_real, fd_substitute, landmarks_real, landmarks_substitute, fraction)
    return dataset
    

if __name__ == "__main__":
    save_path = "data/validation/"
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)

    diagnostic = "NORM"
    lead = 1
    n_beats = 10
    sr = get_sr()
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    trimmed_real_fd, real_landmarks_all = extract_ecg_clinical_landmarks(real_all, n_beats, sr)

    oversmoothing_dataset = oversmoothing_creation(trimmed_real_fd, real_landmarks_all)
    with open(path / "oversmoothing_dataset.pkl", "wb") as f:
        pickle.dump(oversmoothing_dataset, f)
    memorization_dataset = memorization_creation(trimmed_real_fd, trimmed_real_fd, real_landmarks_all, real_landmarks_all)
    with open(path / "memorization_dataset.pkl", "wb") as f:
        pickle.dump(memorization_dataset, f)
    gaussian_noise_dataset = gaussian_noise_creation(trimmed_real_fd, real_landmarks_all)
    with open(path / "gaussian_noise_dataset.pkl", "wb") as f:
        pickle.dump(gaussian_noise_dataset, f)
    mode_collapse_vary_modes_dataset = mode_collapse_vary_modes_creation(trimmed_real_fd, real_landmarks_all)
    with open(path / "mode_collapse_vary_modes_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_modes_dataset, f)
    mode_collapse_vary_spike_ratio_dataset = mode_collapse_vary_spike_ratio_creation(trimmed_real_fd, real_landmarks_all)
    with open(path / "mode_collapse_vary_spike_ratio_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_spike_ratio_dataset, f)
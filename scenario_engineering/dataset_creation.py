import pickle
from pathlib import Path
from scenario_engineering.controlled_flaw_modelling import *
from preprocess.ptbxl_preprocess import get_sr, load_dataset, extract_ecg_phase_aligned, extract_ecg_sliding_windows

def oversmoothing_creation(fd):
    dataset = {}
    window_size = [5, 10, 15, 20, 25]
    for window in window_size:
        dataset[window] = oversmoothing(fd, window)
    return dataset

def memorization_creation(fd_real, fd_substitute):
    dataset = {}
    fraction = [0.1, 0.2, 0.3, 0.4, 0.5]
    for fraction in fraction:
        dataset[fraction] = full_memorization(fd_real, fd_substitute, fraction)
    return dataset

def gaussian_noise_creation(fd):
    dataset = {}
    noise_multiplier = [1.5, 2.0, 2.5, 3.0, 3.5]
    for noise_multiplier in noise_multiplier:
        dataset[noise_multiplier] = gaussian_noise(fd, noise_multiplier)
    return dataset

def mode_collapse_vary_modes_creation(fd):
    dataset = {}
    num_modes = [1, 2, 3, 4, 5]
    for num_modes in num_modes:
        dataset[num_modes] = mode_collapse(fd, num_modes=num_modes)
    return dataset

def mode_collapse_vary_spike_ratio_creation(fd):
    dataset = {}
    spike_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]
    for spike_ratio in spike_ratio:
        dataset[spike_ratio] = mode_collapse(fd, num_modes=1, spike_ratio=spike_ratio)
    return dataset

def segment_leaking_creation(fd_real, fd_substitute):
    dataset = {}
    fraction = [0.1, 0.2, 0.3, 0.4, 0.5]
    for fraction in fraction:
        dataset[fraction] = segment_leaking(fd_real, fd_substitute, fraction)
    return dataset

def phase_shift_creation(fd, landmarks):
    dataset = {}
    shift_fraction = [0.1, 0.2, 0.3, 0.4, 0.5]
    for shift_fraction in shift_fraction:
        dataset[shift_fraction] = phase_shift(fd, landmarks, shift_fraction)
    return dataset

def time_distortion_creation(fd, landmarks):
    dataset = {}
    alpha = [0.5, 1.0, 1.5, 2.0, 2.5]
    for alpha in alpha:
        dataset[alpha] = time_distortion(fd, landmarks, alpha)
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
    # aligned_real_fd = extract_ecg_phase_aligned(real_all, sr)
    # n_sample, n_timepoints, n_channel = aligned_real_fd.data_matrix.shape
    # real_fd = aligned_real_fd[:n_sample//2]
    # substitute_fd = aligned_real_fd[n_sample//2:]

    # oversmoothing_dataset = oversmoothing_creation(real_fd)
    # with open(path / "oversmoothing_dataset.pkl", "wb") as f:
    #     pickle.dump(oversmoothing_dataset, f)
    # memorization_dataset = memorization_creation(real_fd, substitute_fd)
    # with open(path / "memorization_dataset.pkl", "wb") as f:
    #     pickle.dump(memorization_dataset, f)
    # gaussian_noise_dataset = gaussian_noise_creation(real_fd)
    # with open(path / "gaussian_noise_dataset.pkl", "wb") as f:
    #     pickle.dump(gaussian_noise_dataset, f)
    # mode_collapse_vary_modes_dataset = mode_collapse_vary_modes_creation(real_fd)
    # with open(path / "mode_collapse_vary_modes_dataset.pkl", "wb") as f:
    #     pickle.dump(mode_collapse_vary_modes_dataset, f)
    # mode_collapse_vary_spike_ratio_dataset = mode_collapse_vary_spike_ratio_creation(real_fd)
    # with open(path / "mode_collapse_vary_spike_ratio_dataset.pkl", "wb") as f:
    #     pickle.dump(mode_collapse_vary_spike_ratio_dataset, f)
    # segment_leaking_dataset = segment_leaking_creation(real_fd, substitute_fd)
    # with open(path / "segment_leaking_dataset.pkl", "wb") as f:
    #     pickle.dump(segment_leaking_dataset, f)

    segments, landmarks = extract_ecg_sliding_windows(real_all, sr)
    n_data = segments.data_matrix.shape[0]
    real_segments = segments[:n_data//2]
    real_landmarks = landmarks[:n_data//2]
    substitute_segments = segments[n_data//2:]
    substitute_landmarks = landmarks[n_data//2:]

    phase_shift_dataset = phase_shift_creation(real_segments, real_landmarks)
    with open(path / "phase_shift_dataset.pkl", "wb") as f:
        pickle.dump(phase_shift_dataset, f)
    time_distortion_dataset = time_distortion_creation(real_segments, real_landmarks)
    with open(path / "time_distortion_dataset.pkl", "wb") as f:
        pickle.dump(time_distortion_dataset, f)
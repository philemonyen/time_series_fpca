from methods.validation.controlled_flaw_modelling import *

def oversmoothing_creation(fd, landmarks):
    dataset = {}
    window_size = [5, 10, 20, 30, 50, 100]
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
    noise_multiplier = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
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
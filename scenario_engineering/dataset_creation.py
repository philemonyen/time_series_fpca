import pickle
from pathlib import Path
from scenario_engineering.controlled_flaw_modelling import *
from preprocess.ptbxl_preprocess import get_sr, load_dataset, get_landmarks, extract_ecg_sliding_windows

def get_flaw_scales(scenario):
    """
    ECG-calibrated severity ladders for raw PTB-XL strips of shape [N, 1000].

    1000 samples = 10 s x 100 Hz, so 1 sample = 10 ms. At ~75 bpm: RR ~800 ms
    (80 samples), QRS ~80-120 ms (8-12 samples), ~12 beats per strip.
    """
    if scenario == "oversmoothing":
        # Moving-average width in samples (x10 ms). 4 = 40 ms (notch/slur);
        # 8 = 80 ms (narrow QRS); 12 = 120 ms (wide QRS); 16 = 160 ms (ST/T);
        # 20 = 200 ms (T-wave).
        return [4, 8, 12, 16, 20]
    elif scenario == "memorization":
        # Fraction of synthetic records replaced by exact real traces.
        # ECG morphology is identifying; a few clones already leak identity.
        return [0.02, 0.05, 0.10, 0.15, 0.25]
    elif scenario == "gaussian_noise":
        # Noise sigma as a multiple of the ECG std.
        # 0.05 ~ subtle EMG; 0.25 diagnostic quality drops; 1.0 QRS still visible.
        return [0.05, 0.10, 0.25, 0.50, 1.00]
    elif scenario in ("mode_collapse", "mode_collapse_vary_modes"):
        # Number of stereotyped 10 s templates the generator collapses onto.
        return [1, 2, 3, 4, 5]
    elif scenario == "mode_collapse_vary_spike_ratio":
        # Fraction of the synthetic set copied from a single template.
        return [0.05, 0.10, 0.15, 0.20, 0.30]
    elif scenario == "segment_leaking":
        # Fraction of synthetic records that receive a one-beat real splice
        # (~80 samples = 800 ms at 100 Hz).
        return [0.05, 0.10, 0.15, 0.20, 0.30]
    elif scenario == "phase_shift":
        # Delay of internal R-peaks as a fraction of local RR.
        # 0.05 ~ 40 ms; 0.15 ~ 120 ms (QRS-scale); 0.30 ~ 240 ms (marked).
        return [0.05, 0.10, 0.15, 0.20, 0.30]
    elif scenario == "time_distortion":
        # Power-law warp gamma(t)=t^alpha on a 10 s strip. Displacement at
        # mid-record is ~110 ms (0.97/1.03) to ~200 ms (0.94/1.06); 1.00 is identity.
        return [0.94, 0.97, 1.00, 1.03, 1.06]
    else:
        raise ValueError(f"Unknown flaw scenario: {scenario}")

def oversmoothing_creation(data, landmarks):
    dataset = {}
    for window in get_flaw_scales("oversmoothing"):
        dataset[window] = oversmoothing(data, landmarks, window)
    return dataset

def memorization_creation(real_data, substitute_data, real_landmarks, substitute_landmarks):
    dataset = {}
    for fraction in get_flaw_scales("memorization"):
        dataset[fraction] = full_memorization(real_data, substitute_data, real_landmarks, substitute_landmarks, fraction)
    return dataset

def gaussian_noise_creation(data, landmarks):
    dataset = {}
    for noise_multiplier in get_flaw_scales("gaussian_noise"):
        dataset[noise_multiplier] = gaussian_noise(data, landmarks, noise_multiplier)
    return dataset

def mode_collapse_vary_modes_creation(data, landmarks):
    dataset = {}
    for num_modes in get_flaw_scales("mode_collapse_vary_modes"):
        dataset[num_modes] = mode_collapse(data, landmarks, num_modes=num_modes)
    return dataset

def mode_collapse_vary_spike_ratio_creation(data, landmarks):
    dataset = {}
    for spike_ratio in get_flaw_scales("mode_collapse_vary_spike_ratio"):
        dataset[spike_ratio] = mode_collapse(data, landmarks, num_modes=1, spike_ratio=spike_ratio)
    return dataset

def segment_leaking_creation(real_data, substitute_data, real_landmarks, substitute_landmarks):
    dataset = {}
    for fraction in get_flaw_scales("segment_leaking"):
        dataset[fraction] = segment_leaking(
            real_data, substitute_data, real_landmarks, substitute_landmarks, fraction
        )
    return dataset

def phase_shift_creation(data, landmarks):
    dataset = {}
    for shift_fraction in get_flaw_scales("phase_shift"):
        dataset[shift_fraction] = phase_shift(data, landmarks, shift_fraction)
    return dataset

def time_distortion_creation(data, landmarks):
    dataset = {}
    for alpha in get_flaw_scales("time_distortion"):
        dataset[alpha] = time_distortion(data, landmarks, alpha)
    return dataset

if __name__ == "__main__":
    save_path = "data/validation/"
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)

    diagnostic = "NORM"
    lead = 1
    n_beats = 10
    sr = get_sr()
    
    ### Morphological Flawed Dataset Creation ###
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    landmarks = get_landmarks(real_all, sr)
    
    n_data = real_all.shape[0]
    real_data = real_all[:n_data//2]
    real_landmarks = landmarks[:n_data//2]
    substitute_data = real_all[n_data//2:]
    substitute_landmarks = landmarks[n_data//2:]

    with open(path / "real_fd.pkl", "wb") as f:
        pickle.dump(real_data, f)
    with open(path / "substitute_fd.pkl", "wb") as f:
        pickle.dump(substitute_data, f)
    oversmoothing_dataset = oversmoothing_creation(real_data, real_landmarks)
    with open(path / "oversmoothing_dataset.pkl", "wb") as f:
        pickle.dump(oversmoothing_dataset, f)
    memorization_dataset = memorization_creation(real_data, substitute_data, landmarks, landmarks)
    with open(path / "memorization_dataset.pkl", "wb") as f:
        pickle.dump(memorization_dataset, f)
    gaussian_noise_dataset = gaussian_noise_creation(real_data, real_landmarks)
    with open(path / "gaussian_noise_dataset.pkl", "wb") as f:
        pickle.dump(gaussian_noise_dataset, f)
    mode_collapse_vary_modes_dataset = mode_collapse_vary_modes_creation(real_data, real_landmarks)
    with open(path / "mode_collapse_vary_modes_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_modes_dataset, f)
    mode_collapse_vary_spike_ratio_dataset = mode_collapse_vary_spike_ratio_creation(real_data, real_landmarks)
    with open(path / "mode_collapse_vary_spike_ratio_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_spike_ratio_dataset, f)
    segment_leaking_dataset = segment_leaking_creation(real_data, substitute_data, real_landmarks, substitute_landmarks)
    with open(path / "segment_leaking_dataset.pkl", "wb") as f:
        pickle.dump(segment_leaking_dataset, f)

    ### Temporal Flawed Dataset Creation ###
    phase_shift_dataset = phase_shift_creation(real_data, real_landmarks)
    with open(path / "phase_shift_dataset.pkl", "wb") as f:
        pickle.dump(phase_shift_dataset, f)
    time_distortion_dataset = time_distortion_creation(real_data, real_landmarks)
    with open(path / "time_distortion_dataset.pkl", "wb") as f:
        pickle.dump(time_distortion_dataset, f)
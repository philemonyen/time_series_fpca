import pickle
from pathlib import Path
from scenario_engineering.controlled_flaw_modelling import *
from preprocess.ptbxl_preprocess import get_sr, load_dataset, get_landmarks, align_ecg, extract_ecg_sliding_windows

def get_morphology_scenarios():
    return [
        "oversmoothing", 
        "gaussian_noise",
        "baseline_drift", 
        "spurious_transient"
    ]

def get_temporal_scenarios():
    return [
        "phase_shift", 
        "time_distortion", 
        "phase_jitter", 
        "loss_of_autocorrelation", 
    ]

def get_distributional_scenarios():
    return [
        "mode_collapse_vary_modes", 
        "mode_collapse_vary_spike_ratio", 
    ]

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
        
    elif scenario == "mode_collapse_vary_modes":
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
        
    elif scenario == "phase_jitter":
        # Standard deviation of the random shift per beat as a fraction of local RR.
        # 0.02 ~ subtle 16 ms jitter; 0.05 ~ 40 ms; 0.20 ~ 160 ms (highly irregular).
        return [0.02, 0.05, 0.10, 0.15, 0.20]
        
    elif scenario == "baseline_drift":
        # Amplitude of the low-frequency drift wave as a fraction of the signal's std.
        # 0.10 ~ subtle wander; 1.00 ~ drift amplitude equals the signal's variance.
        return [0.10, 0.25, 0.50, 0.75, 1.00]
        
    elif scenario == "loss_of_autocorrelation":
        # Fraction of internal RR segments (heartbeats) selected for random shuffling.
        # 0.20 ~ ~2 beats swapped (minor temporal glitch); 1.00 ~ fully randomized sequence.
        return [0.20, 0.40, 0.60, 0.80, 1.00]
        
    elif scenario == "spurious_transient":
        # Amplitude of a hallucinated, high-frequency spike as a multiple of signal std.
        # 0.5 ~ minor P/T-wave sized notch; 2.0 ~ prominent artifact; 4.0 ~ massive, unnatural spike.
        return [0.5, 1.0, 2.0, 3.0, 4.0]
        
    else:
        raise ValueError(f"Unknown flaw scenario: {scenario}")

# Morphological Flawed Dataset Creation 
def oversmoothing_creation(data, landmarks):
    dataset = {}
    for window in get_flaw_scales("oversmoothing"):
        flaw_data, flaw_landmarks = oversmoothing(data, landmarks, window)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        dataset[window] = flaw_data, align_fd
    return dataset

def gaussian_noise_creation(data, landmarks):
    dataset = {}
    for noise_multiplier in get_flaw_scales("gaussian_noise"):
        flaw_data, flaw_landmarks = gaussian_noise(data, landmarks, noise_multiplier)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        dataset[noise_multiplier] = flaw_data, align_fd
    return dataset

def baseline_drift_creation(data, landmarks):
    dataset = {}
    for amplitude_fraction in get_flaw_scales("baseline_drift"):
        flaw_data, flaw_landmarks = baseline_drift(data, landmarks, amplitude_fraction=amplitude_fraction)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        dataset[amplitude_fraction] = flaw_data, align_fd
    return dataset

def spurious_transient_creation(data, landmarks):
    dataset = {}
    for amplitude_fraction in get_flaw_scales("spurious_transient"):
        flaw_data, flaw_landmarks = spurious_transient(data, landmarks, amplitude_fraction=amplitude_fraction)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        dataset[amplitude_fraction] = flaw_data, align_fd
    return dataset

# Distributional Flawed Dataset Creation 
def mode_collapse_vary_modes_creation(data, landmarks):
    dataset = {}
    for num_modes in get_flaw_scales("mode_collapse_vary_modes"):
        flaw_data, flaw_landmarks = mode_collapse(data, landmarks, num_modes=num_modes)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        dataset[num_modes] = flaw_data, align_fd
    return dataset

def mode_collapse_vary_spike_ratio_creation(data, landmarks):
    dataset = {}
    for spike_ratio in get_flaw_scales("mode_collapse_vary_spike_ratio"):
        flaw_data, flaw_landmarks = mode_collapse(data, landmarks, num_modes=1, spike_ratio=spike_ratio)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        dataset[spike_ratio] = flaw_data, align_fd
    return dataset

# Temporal Flawed Dataset Creation 
def phase_shift_creation(data, landmarks):
    dataset = {}
    for shift_fraction in get_flaw_scales("phase_shift"):
        flaw_data, flaw_landmarks = phase_shift(data, landmarks, shift_fraction=shift_fraction)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        dataset[shift_fraction] = flaw_data,flaw_segments, segment_landmarks
    return dataset

def time_distortion_creation(data, landmarks):
    dataset = {}
    for alpha in get_flaw_scales("time_distortion"):
        flaw_data, flaw_landmarks = time_distortion(data, landmarks, alpha=alpha)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        dataset[alpha] = flaw_data, flaw_segments, segment_landmarks
    return dataset

def phase_jitter_creation(data, landmarks):
    dataset = {}
    for jitter_fraction in get_flaw_scales("phase_jitter"):
        flaw_data, flaw_landmarks = phase_jitter(data, landmarks, jitter_fraction=jitter_fraction)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        dataset[jitter_fraction] = flaw_data, flaw_segments, segment_landmarks
    return dataset

def loss_of_autocorrelation_creation(data, landmarks):
    dataset = {}
    for shuffle_ratio in get_flaw_scales("loss_of_autocorrelation"):
        flaw_data, flaw_landmarks = loss_of_autocorrelation(data, landmarks, shuffle_ratio=shuffle_ratio)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        dataset[shuffle_ratio] = flaw_data, flaw_segments, segment_landmarks
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
    landmarks = get_landmarks(real_all, sr)
    
    n_data = real_all.shape[0]
    real_data = real_all[:n_data//2]
    real_landmarks = landmarks[:n_data//2]
    substitute_data = real_all[n_data//2:]
    substitute_landmarks = landmarks[n_data//2:]

    # Original Unaligned Dataset
    with open(path / "real_data.pkl", "wb") as f:
        pickle.dump(real_data, f)
    with open(path / "substitute_data.pkl", "wb") as f:
        pickle.dump(substitute_data, f)
    with open(path / "substitute_landmarks.pkl", "wb") as f:
        pickle.dump(substitute_landmarks, f)

    # Original Aligned Dataset
    read_fd = align_ecg(real_data, real_landmarks)
    with open(path / "real_fd.pkl", "wb") as f:
        pickle.dump(read_fd, f)
    substitute_fd = align_ecg(substitute_data, substitute_landmarks)
    with open(path / "substitute_fd.pkl", "wb") as f:
        pickle.dump(substitute_fd, f)

    # Morphological Flawed Dataset Creation
    oversmoothing_dataset = oversmoothing_creation(real_data, real_landmarks)
    with open(path / "oversmoothing_dataset.pkl", "wb") as f:
        pickle.dump(oversmoothing_dataset, f)
    gaussian_noise_dataset = gaussian_noise_creation(real_data, real_landmarks)
    with open(path / "gaussian_noise_dataset.pkl", "wb") as f:
        pickle.dump(gaussian_noise_dataset, f)
    baseline_drift_dataset = baseline_drift_creation(real_data, real_landmarks)
    with open(path / "baseline_drift_dataset.pkl", "wb") as f:
        pickle.dump(baseline_drift_dataset, f)
    spurious_transient_dataset = spurious_transient_creation(real_data, real_landmarks)
    with open(path / "spurious_transient_dataset.pkl", "wb") as f:
        pickle.dump(spurious_transient_dataset, f)

    # Distributional Flawed Dataset Creation
    mode_collapse_vary_modes_dataset = mode_collapse_vary_modes_creation(real_data, real_landmarks)
    with open(path / "mode_collapse_vary_modes_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_modes_dataset, f)
    mode_collapse_vary_spike_ratio_dataset = mode_collapse_vary_spike_ratio_creation(real_data, real_landmarks)
    with open(path / "mode_collapse_vary_spike_ratio_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_spike_ratio_dataset, f)

    # Temporal Flawed Dataset Creation
    # Keep the original P/R/T table for the flaw models; sliding-window
    # landmarks are relative R-peak locations and are not a drop-in replacement.
    real_segments, real_segment_landmarks = extract_ecg_sliding_windows(real_data, real_landmarks)
    with open(path / "real_segments.pkl", "wb") as f:
        pickle.dump((real_segments, real_segment_landmarks), f)
    phase_shift_dataset = phase_shift_creation(real_data, real_landmarks)
    with open(path / "phase_shift_dataset.pkl", "wb") as f:
        pickle.dump(phase_shift_dataset, f)
    time_distortion_dataset = time_distortion_creation(real_data, real_landmarks)
    with open(path / "time_distortion_dataset.pkl", "wb") as f:
        pickle.dump(time_distortion_dataset, f)
    phase_jitter_dataset = phase_jitter_creation(real_data, real_landmarks)
    with open(path / "phase_jitter_dataset.pkl", "wb") as f:
        pickle.dump(phase_jitter_dataset, f)
    loss_of_autocorrelation_dataset = loss_of_autocorrelation_creation(real_data, real_landmarks)
    with open(path / "loss_of_autocorrelation_dataset.pkl", "wb") as f:
        pickle.dump(loss_of_autocorrelation_dataset, f)
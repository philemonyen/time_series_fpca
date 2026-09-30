import pickle
from pathlib import Path
from scenario_engineering.controlled_flaw_modelling import *
from preprocess.ptbxl_preprocess import get_sr, load_dataset, get_landmarks, align_ecg, extract_ecg_sliding_windows
from preprocess.ecg_clinical_preprocess import get_pqrst
from transformation.fda.fpca import fpca_with_param, basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from skfda import FDataGrid
from sklearn.model_selection import train_test_split

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
        
    elif scenario == "gaussian_noise":
        # Noise sigma as a multiple of the ECG std.
        # 0.05 ~ subtle EMG; 0.25 diagnostic quality drops; 1.0 QRS still visible.
        return [0.05, 0.10, 0.25, 0.50, 1.00]
        
    elif scenario == "baseline_drift":
        # Amplitude of the low-frequency drift wave as a fraction of the signal's std.
        # 0.10 ~ subtle wander; 1.00 ~ drift amplitude equals the signal's variance.
        return [0.10, 0.25, 0.50, 0.75, 1.00]
        
    elif scenario == "spurious_transient":
        # Amplitude of a hallucinated, high-frequency spike as a multiple of signal std.
        # 0.5 ~ minor P/T-wave sized notch; 2.0 ~ prominent artifact; 4.0 ~ massive, unnatural spike.
        return [0.5, 1.0, 2.0, 3.0, 4.0]
        
    elif scenario == "phase_shift":
        # Delay of internal R-peaks as a fraction of local RR.
        # 0.05 ~ 40 ms; 0.15 ~ 120 ms (QRS-scale); 0.30 ~ 240 ms (marked).
        return [0.05, 0.10, 0.15, 0.20, 0.30]
        
    elif scenario == "time_distortion":
        # Power-law warp gamma(t)=t^alpha on a 10 s strip. Displacement at
        # mid-record is ~110 ms (0.97/1.03) to ~200 ms (0.94/1.06); 1.00 is identity.
        return [1.00, 1.03, 1.06, 1.09, 1.12]
        
    elif scenario == "phase_jitter":
        # Standard deviation of the random shift per beat as a fraction of local RR.
        # 0.02 ~ subtle 16 ms jitter; 0.05 ~ 40 ms; 0.20 ~ 160 ms (highly irregular).
        return [0.02, 0.05, 0.10, 0.15, 0.20]
        
    elif scenario == "loss_of_autocorrelation":
        # Fraction of internal RR segments (heartbeats) selected for random shuffling.
        # 0.20 ~ ~2 beats swapped (minor temporal glitch); 1.00 ~ fully randomized sequence.
        return [0.20, 0.40, 0.60, 0.80, 1.00]
        
    elif scenario == "mode_collapse_vary_modes":
        # Number of stereotyped 10 s templates the generator collapses onto.
        return [1, 2, 3, 4, 5]
        
    elif scenario == "mode_collapse_vary_spike_ratio":
        # Fraction of the synthetic set copied from a single template.
        return [0.2, 0.3, 0.4, 0.5, 0.6]
        
    elif scenario == "segment_leaking":
        # Fraction of synthetic records that receive a one-beat real splice
        # (~80 samples = 800 ms at 100 Hz).
        return [0.05, 0.10, 0.15, 0.20, 0.30]
        
    else:
        raise ValueError(f"Unknown flaw scenario: {scenario}")

# Morphological Flawed Dataset Creation 
def oversmoothing_creation(data, landmarks, clinical_landmarks):
    morphological_eval = {}
    temporal_eval = {}
    clinical_eval = {}
    for window in get_flaw_scales("oversmoothing"):
        flaw_data, flaw_landmarks, flaw_clinical_landmarks = oversmoothing(data, landmarks, clinical_landmarks, window)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[window] = flaw_data, align_fd
        temporal_eval[window] = flaw_data, flaw_segments, segment_landmarks
        clinical_eval[window] = flaw_data, flaw_clinical_landmarks
    return morphological_eval, temporal_eval, clinical_eval

def gaussian_noise_creation(data, landmarks, clinical_landmarks):
    morphological_eval = {}
    temporal_eval = {}
    clinical_eval = {}
    for noise_multiplier in get_flaw_scales("gaussian_noise"):
        flaw_data, flaw_landmarks, flaw_clinical_landmarks = gaussian_noise(data, landmarks, clinical_landmarks, noise_multiplier)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[noise_multiplier] = flaw_data, align_fd
        temporal_eval[noise_multiplier] = flaw_data, flaw_segments, segment_landmarks
        clinical_eval[noise_multiplier] = flaw_data, flaw_clinical_landmarks
    return morphological_eval, temporal_eval, clinical_eval

def baseline_drift_creation(data, landmarks, clinical_landmarks):
    morphological_eval = {}
    temporal_eval = {}
    clinical_eval = {}
    for amplitude_fraction in get_flaw_scales("baseline_drift"):
        flaw_data, flaw_landmarks, flaw_clinical_landmarks = baseline_drift(data, landmarks, clinical_landmarks, amplitude_fraction=amplitude_fraction)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[amplitude_fraction] = flaw_data, align_fd
        temporal_eval[amplitude_fraction] = flaw_data, flaw_segments, segment_landmarks
        clinical_eval[amplitude_fraction] = flaw_data, flaw_clinical_landmarks
    return morphological_eval, temporal_eval, clinical_eval

def spurious_transient_creation(data, landmarks, clinical_landmarks):
    morphological_eval = {}
    temporal_eval = {}
    clinical_eval = {}
    for amplitude_fraction in get_flaw_scales("spurious_transient"):
        flaw_data, flaw_landmarks, flaw_clinical_landmarks = spurious_transient(data, landmarks, clinical_landmarks, amplitude_fraction=amplitude_fraction)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[amplitude_fraction] = flaw_data, align_fd
        temporal_eval[amplitude_fraction] = flaw_data, flaw_segments, segment_landmarks
        clinical_eval[amplitude_fraction] = flaw_data, flaw_clinical_landmarks
    return morphological_eval, temporal_eval, clinical_eval

# Distributional Flawed Dataset Creation 
def mode_collapse_vary_modes_creation(data, landmarks, clinical_landmarks):
    morphological_eval = {}
    temporal_eval = {}
    clinical_eval = {}
    for num_modes in get_flaw_scales("mode_collapse_vary_modes"):
        flaw_data, flaw_landmarks, flaw_clinical_landmarks = mode_collapse(data, landmarks, clinical_landmarks, num_modes=num_modes)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[num_modes] = flaw_data, align_fd
        temporal_eval[num_modes] = flaw_data, flaw_segments, segment_landmarks
        clinical_eval[num_modes] = flaw_data, flaw_clinical_landmarks
    return morphological_eval, temporal_eval, clinical_eval

def mode_collapse_vary_spike_ratio_creation(data, landmarks, clinical_landmarks):
    morphological_eval = {}
    temporal_eval = {}
    clinical_eval = {}
    for spike_ratio in get_flaw_scales("mode_collapse_vary_spike_ratio"):
        flaw_data, flaw_landmarks, flaw_clinical_landmarks = mode_collapse(data, landmarks, clinical_landmarks, num_modes=1, spike_ratio=spike_ratio)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[spike_ratio] = flaw_data, align_fd
        temporal_eval[spike_ratio] = flaw_data, flaw_segments, segment_landmarks
        clinical_eval[spike_ratio] = flaw_data, flaw_clinical_landmarks
    return morphological_eval, temporal_eval, clinical_eval

# Temporal Flawed Dataset Creation 
def phase_shift_creation(data, landmarks, clinical_landmarks):
    morphological_eval = {}
    temporal_eval = {}
    clinical_eval = {}
    for shift_fraction in get_flaw_scales("phase_shift"):
        flaw_data, flaw_landmarks, flaw_clinical_landmarks = phase_shift(data, landmarks, clinical_landmarks, shift_fraction=shift_fraction)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[shift_fraction] = flaw_data, align_fd
        temporal_eval[shift_fraction] = flaw_data, flaw_segments, segment_landmarks
        clinical_eval[shift_fraction] = flaw_data, flaw_clinical_landmarks
    return morphological_eval, temporal_eval, clinical_eval

def time_distortion_creation(data, landmarks, clinical_landmarks):
    morphological_eval = {}
    temporal_eval = {}
    clinical_eval = {}
    for alpha in get_flaw_scales("time_distortion"):
        flaw_data, flaw_landmarks, flaw_clinical_landmarks = time_distortion(data, landmarks, clinical_landmarks, alpha=alpha)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[alpha] = flaw_data, align_fd
        temporal_eval[alpha] = flaw_data, flaw_segments, segment_landmarks
        clinical_eval[alpha] = flaw_data, flaw_clinical_landmarks
    return morphological_eval, temporal_eval, clinical_eval

def phase_jitter_creation(data, landmarks, clinical_landmarks):
    morphological_eval = {}
    temporal_eval = {}
    clinical_eval = {}
    for jitter_fraction in get_flaw_scales("phase_jitter"):
        flaw_data, flaw_landmarks, flaw_clinical_landmarks = phase_jitter(data, landmarks, clinical_landmarks, jitter_fraction=jitter_fraction)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[jitter_fraction] = flaw_data, align_fd
        temporal_eval[jitter_fraction] = flaw_data, flaw_segments, segment_landmarks
        clinical_eval[jitter_fraction] = flaw_data, flaw_clinical_landmarks
    return morphological_eval, temporal_eval, clinical_eval

def loss_of_autocorrelation_creation(data, landmarks, clinical_landmarks):
    morphological_eval = {}
    temporal_eval = {}
    clinical_eval = {}
    for shuffle_ratio in get_flaw_scales("loss_of_autocorrelation"):
        flaw_data, flaw_landmarks, flaw_clinical_landmarks = loss_of_autocorrelation(data, landmarks, clinical_landmarks, shuffle_ratio=shuffle_ratio)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[shuffle_ratio] = flaw_data, align_fd
        temporal_eval[shuffle_ratio] = flaw_data, flaw_segments, segment_landmarks
        clinical_eval[shuffle_ratio] = flaw_data, flaw_clinical_landmarks
    return morphological_eval, temporal_eval, clinical_eval

if __name__ == "__main__":
    real_path = "data/validation/"
    morphology_save_path = "data/validation/morphology/"
    temporal_save_path = "data/validation/temporal/"
    clinical_save_path = "data/validation/clinical/"
    save_path = Path(real_path)
    morphology_path = Path(morphology_save_path)
    temporal_path = Path(temporal_save_path)
    clinical_path = Path(clinical_save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    morphology_path.mkdir(parents=True, exist_ok=True)
    temporal_path.mkdir(parents=True, exist_ok=True)
    clinical_path.mkdir(parents=True, exist_ok=True)

    diagnostic = "NORM"
    lead = 1
    n_beats = 10
    sr = get_sr()
    
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    landmarks = get_landmarks(real_all, sr)
    clinical_landmarks = get_pqrst(real_all, sr)
    
    n_data = real_all.shape[0]
    real_data = real_all[:n_data//2]
    real_landmarks = landmarks[:n_data//2]
    real_clinical_landmarks = clinical_landmarks[:n_data//2]
    substitute_data = real_all[n_data//2:]
    substitute_landmarks = landmarks[n_data//2:]
    substitute_clinical_landmarks = clinical_landmarks[n_data//2:]

    # Original Unaligned and AlignedDataset
    with open(save_path / "real_data.pkl", "wb") as f:
        pickle.dump(real_data, f)
    real_fd = align_ecg(real_data, real_landmarks)
    with open(save_path / "real_fd.pkl", "wb") as f:
        pickle.dump(real_fd, f)
    with open(clinical_path / "real_clinical_landmarks.pkl", "wb") as f:
        pickle.dump(real_clinical_landmarks, f)

    # Original Segments 
    real_segments, real_segment_landmarks = extract_ecg_sliding_windows(real_data, real_landmarks)
    with open(save_path / "real_segments.pkl", "wb") as f:
        pickle.dump((real_segments, real_segment_landmarks), f)

    # Substitue Unaligned and Aligned Dataset (For Privacy Evaluation)
    with open(save_path / "substitute_data.pkl", "wb") as f:
        pickle.dump(substitute_data, f)
    substitute_fd = align_ecg(substitute_data, substitute_landmarks)
    with open(save_path / "substitute_fd.pkl", "wb") as f:
        pickle.dump(substitute_fd, f)

    # Morphological Flawed Dataset Creation
    oversmoothing_morphology_dataset, oversmoothing_temporal_dataset, oversmoothing_clinical_dataset = oversmoothing_creation(real_data, real_landmarks, real_clinical_landmarks)
    with open(morphology_path / "oversmoothing_dataset.pkl", "wb") as f:
        pickle.dump(oversmoothing_morphology_dataset, f)
    with open(temporal_path / "oversmoothing_dataset.pkl", "wb") as f:
        pickle.dump(oversmoothing_temporal_dataset, f)
    with open(clinical_path / "oversmoothing_clinical_dataset.pkl", "wb") as f:
        pickle.dump(oversmoothing_clinical_dataset, f)
    gaussian_noise_morphology_dataset, gaussian_noise_temporal_dataset, gaussian_noise_clinical_dataset = gaussian_noise_creation(real_data, real_landmarks, real_clinical_landmarks)
    with open(morphology_path / "gaussian_noise_dataset.pkl", "wb") as f:
        pickle.dump(gaussian_noise_morphology_dataset, f)
    with open(temporal_path / "gaussian_noise_dataset.pkl", "wb") as f:
        pickle.dump(gaussian_noise_temporal_dataset, f)
    with open(clinical_path / "gaussian_noise_clinical_dataset.pkl", "wb") as f:
        pickle.dump(gaussian_noise_clinical_dataset, f)
    baseline_drift_morphology_dataset, baseline_drift_temporal_dataset, baseline_drift_clinical_dataset = baseline_drift_creation(real_data, real_landmarks, real_clinical_landmarks)
    with open(morphology_path / "baseline_drift_dataset.pkl", "wb") as f:
        pickle.dump(baseline_drift_morphology_dataset, f)
    with open(temporal_path / "baseline_drift_dataset.pkl", "wb") as f:
        pickle.dump(baseline_drift_temporal_dataset, f)
    with open(clinical_path / "baseline_drift_clinical_dataset.pkl", "wb") as f:
        pickle.dump(baseline_drift_clinical_dataset, f)
    spurious_transient_morphology_dataset, spurious_transient_temporal_dataset, spurious_transient_clinical_dataset = spurious_transient_creation(real_data, real_landmarks, real_clinical_landmarks)
    with open(morphology_path / "spurious_transient_dataset.pkl", "wb") as f:
        pickle.dump(spurious_transient_morphology_dataset, f)
    with open(temporal_path / "spurious_transient_dataset.pkl", "wb") as f:
        pickle.dump(spurious_transient_temporal_dataset, f)
    with open(clinical_path / "spurious_transient_clinical_dataset.pkl", "wb") as f:
        pickle.dump(spurious_transient_clinical_dataset, f)

    # Distributional Flawed Dataset Creation
    mode_collapse_vary_modes_morphology_dataset, mode_collapse_vary_modes_temporal_dataset, mode_collapse_vary_modes_clinical_dataset = mode_collapse_vary_modes_creation(real_data, real_landmarks, real_clinical_landmarks)
    with open(morphology_path / "mode_collapse_vary_modes_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_modes_morphology_dataset, f)
    with open(temporal_path / "mode_collapse_vary_modes_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_modes_temporal_dataset, f)
    with open(clinical_path / "mode_collapse_vary_modes_clinical_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_modes_clinical_dataset, f)
    mode_collapse_vary_spike_ratio_morphology_dataset, mode_collapse_vary_spike_ratio_temporal_dataset, mode_collapse_vary_spike_ratio_clinical_dataset = mode_collapse_vary_spike_ratio_creation(real_data, real_landmarks, real_clinical_landmarks)
    with open(morphology_path / "mode_collapse_vary_spike_ratio_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_spike_ratio_morphology_dataset, f)
    with open(temporal_path / "mode_collapse_vary_spike_ratio_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_spike_ratio_temporal_dataset, f)
    with open(clinical_path / "mode_collapse_vary_spike_ratio_clinical_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_spike_ratio_clinical_dataset, f)

    # Temporal Flawed Dataset Creation
    phase_shift_morphology_dataset, phase_shift_temporal_dataset, phase_shift_clinical_dataset = phase_shift_creation(real_data, real_landmarks, real_clinical_landmarks)
    with open(morphology_path / "phase_shift_dataset.pkl", "wb") as f:
        pickle.dump(phase_shift_morphology_dataset, f)
    with open(temporal_path / "phase_shift_dataset.pkl", "wb") as f:
        pickle.dump(phase_shift_temporal_dataset, f)
    with open(clinical_path / "phase_shift_clinical_dataset.pkl", "wb") as f:
        pickle.dump(phase_shift_clinical_dataset, f)
    time_distortion_morphology_dataset, time_distortion_temporal_dataset, time_distortion_clinical_dataset = time_distortion_creation(real_data, real_landmarks, real_clinical_landmarks)
    with open(morphology_path / "time_distortion_dataset.pkl", "wb") as f:
        pickle.dump(time_distortion_morphology_dataset, f)
    with open(temporal_path / "time_distortion_dataset.pkl", "wb") as f:
        pickle.dump(time_distortion_temporal_dataset, f)
    with open(clinical_path / "time_distortion_clinical_dataset.pkl", "wb") as f:
        pickle.dump(time_distortion_clinical_dataset, f)
    phase_jitter_morphology_dataset, phase_jitter_temporal_dataset, phase_jitter_clinical_dataset = phase_jitter_creation(real_data, real_landmarks, real_clinical_landmarks)
    with open(morphology_path / "phase_jitter_dataset.pkl", "wb") as f:
        pickle.dump(phase_jitter_morphology_dataset, f)
    with open(temporal_path / "phase_jitter_dataset.pkl", "wb") as f:
        pickle.dump(phase_jitter_temporal_dataset, f)
    with open(clinical_path / "phase_jitter_clinical_dataset.pkl", "wb") as f:
        pickle.dump(phase_jitter_clinical_dataset, f)
    loss_of_autocorrelation_morphology_dataset, loss_of_autocorrelation_temporal_dataset, loss_of_autocorrelation_clinical_dataset = loss_of_autocorrelation_creation(real_data, real_landmarks, real_clinical_landmarks)
    with open(morphology_path / "loss_of_autocorrelation_dataset.pkl", "wb") as f:
        pickle.dump(loss_of_autocorrelation_morphology_dataset, f)
    with open(temporal_path / "loss_of_autocorrelation_dataset.pkl", "wb") as f:
        pickle.dump(loss_of_autocorrelation_temporal_dataset, f)
    with open(clinical_path / "loss_of_autocorrelation_clinical_dataset.pkl", "wb") as f:
        pickle.dump(loss_of_autocorrelation_clinical_dataset, f)
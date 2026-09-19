import pickle
from pathlib import Path
from scenario_engineering.controlled_flaw_modelling import *
from preprocess.ptbxl_preprocess import get_sr, load_dataset, get_landmarks, align_ecg, extract_ecg_sliding_windows
from preprocess.fpca_preprocess import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, landmark_registration
from transformation.fda.fpca import fpca_with_param
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
        return [0.94, 0.97, 1.00, 1.03, 1.06]
        
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
def oversmoothing_creation(data, landmarks):
    morphological_eval = {}
    temporal_eval = {}
    for window in get_flaw_scales("oversmoothing"):
        flaw_data, flaw_landmarks = oversmoothing(data, landmarks, window)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[window] = flaw_data, align_fd
        temporal_eval[window] = flaw_data, flaw_segments, segment_landmarks
    return morphological_eval, temporal_eval

def gaussian_noise_creation(data, landmarks):
    morphological_eval = {}
    temporal_eval = {}
    for noise_multiplier in get_flaw_scales("gaussian_noise"):
        flaw_data, flaw_landmarks = gaussian_noise(data, landmarks, noise_multiplier)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[noise_multiplier] = flaw_data, align_fd
        temporal_eval[noise_multiplier] = flaw_data, flaw_segments, segment_landmarks
    return morphological_eval, temporal_eval

def baseline_drift_creation(data, landmarks):
    morphological_eval = {}
    temporal_eval = {}
    for amplitude_fraction in get_flaw_scales("baseline_drift"):
        flaw_data, flaw_landmarks = baseline_drift(data, landmarks, amplitude_fraction=amplitude_fraction)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[amplitude_fraction] = flaw_data, align_fd
        temporal_eval[amplitude_fraction] = flaw_data, flaw_segments, segment_landmarks
    return morphological_eval, temporal_eval

def spurious_transient_creation(data, landmarks):
    morphological_eval = {}
    temporal_eval = {}
    for amplitude_fraction in get_flaw_scales("spurious_transient"):
        flaw_data, flaw_landmarks = spurious_transient(data, landmarks, amplitude_fraction=amplitude_fraction)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[amplitude_fraction] = flaw_data, align_fd
        temporal_eval[amplitude_fraction] = flaw_data, flaw_segments, segment_landmarks
    return morphological_eval, temporal_eval

# Distributional Flawed Dataset Creation 
def mode_collapse_vary_modes_creation(data, landmarks):
    morphological_eval = {}
    temporal_eval = {}
    for num_modes in get_flaw_scales("mode_collapse_vary_modes"):
        flaw_data, flaw_landmarks = mode_collapse(data, landmarks, num_modes=num_modes)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[num_modes] = flaw_data, align_fd
        temporal_eval[num_modes] = flaw_data, flaw_segments, segment_landmarks
    return morphological_eval, temporal_eval

def mode_collapse_vary_spike_ratio_creation(data, landmarks):
    morphological_eval = {}
    temporal_eval = {}
    for spike_ratio in get_flaw_scales("mode_collapse_vary_spike_ratio"):
        flaw_data, flaw_landmarks = mode_collapse(data, landmarks, num_modes=1, spike_ratio=spike_ratio)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[spike_ratio] = flaw_data, align_fd
        temporal_eval[spike_ratio] = flaw_data, flaw_segments, segment_landmarks
    return morphological_eval, temporal_eval

# Temporal Flawed Dataset Creation 
def phase_shift_creation(data, landmarks):
    morphological_eval = {}
    temporal_eval = {}
    for shift_fraction in get_flaw_scales("phase_shift"):
        flaw_data, flaw_landmarks = phase_shift(data, landmarks, shift_fraction=shift_fraction)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[shift_fraction] = flaw_data, align_fd
        temporal_eval[shift_fraction] = flaw_data, flaw_segments, segment_landmarks
    return morphological_eval, temporal_eval

def time_distortion_creation(data, landmarks):
    morphological_eval = {}
    temporal_eval = {}
    for alpha in get_flaw_scales("time_distortion"):
        flaw_data, flaw_landmarks = time_distortion(data, landmarks, alpha=alpha)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[alpha] = flaw_data, align_fd
        temporal_eval[alpha] = flaw_data, flaw_segments, segment_landmarks
    return morphological_eval, temporal_eval

def phase_jitter_creation(data, landmarks):
    morphological_eval = {}
    temporal_eval = {}
    for jitter_fraction in get_flaw_scales("phase_jitter"):
        flaw_data, flaw_landmarks = phase_jitter(data, landmarks, jitter_fraction=jitter_fraction)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[jitter_fraction] = flaw_data, align_fd
        temporal_eval[jitter_fraction] = flaw_data, flaw_segments, segment_landmarks
    return morphological_eval, temporal_eval

def loss_of_autocorrelation_creation(data, landmarks):
    morphological_eval = {}
    temporal_eval = {}
    for shuffle_ratio in get_flaw_scales("loss_of_autocorrelation"):
        flaw_data, flaw_landmarks = loss_of_autocorrelation(data, landmarks, shuffle_ratio=shuffle_ratio)
        align_fd = align_ecg(flaw_data, flaw_landmarks)
        flaw_segments, segment_landmarks = extract_ecg_sliding_windows(flaw_data, flaw_landmarks)
        morphological_eval[shuffle_ratio] = flaw_data, align_fd
        temporal_eval[shuffle_ratio] = flaw_data, flaw_segments, segment_landmarks
    return morphological_eval, temporal_eval

def privacy_leaking_creation(real_fpcs, real_scores, eigenvalues):
    portion_ratios = [0.1, 0.15, 0.2, 0.25, 0.3]
    leaking_ratios = [0.5, 1, 1.5, 2, 2.5]
    dataset = {}
    for portion_ratio in portion_ratios:
        for leaking_ratio in leaking_ratios:
            synthetic_data, compromised_scores = privacy_leaking_data(real_fpcs, real_scores, eigenvalues, portion_ratio, leaking_ratio)
            dataset[portion_ratio, leaking_ratio] = synthetic_data, compromised_scores
    return dataset

if __name__ == "__main__":
    real_path = "data/validation/"
    morphology_save_path = "data/validation/morphology/"
    temporal_save_path = "data/validation/temporal/"
    privacy_save_path = "data/validation/privacy/"
    save_path = Path(real_path)
    morphology_path = Path(morphology_save_path)
    temporal_path = Path(temporal_save_path)
    privacy_path = Path(privacy_save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    morphology_path.mkdir(parents=True, exist_ok=True)
    temporal_path.mkdir(parents=True, exist_ok=True)
    privacy_path.mkdir(parents=True, exist_ok=True)
    
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

    # Original Unaligned and AlignedDataset
    with open(save_path / "real_data.pkl", "wb") as f:
        pickle.dump(real_data, f)
    real_fd = align_ecg(real_data, real_landmarks)
    with open(save_path / "real_fd.pkl", "wb") as f:
        pickle.dump(real_fd, f)

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
    oversmoothing_morphology_dataset, oversmoothing_temporal_dataset = oversmoothing_creation(real_data, real_landmarks)
    with open(morphology_path / "oversmoothing_dataset.pkl", "wb") as f:
        pickle.dump(oversmoothing_morphology_dataset, f)
    with open(temporal_path / "oversmoothing_dataset.pkl", "wb") as f:
        pickle.dump(oversmoothing_temporal_dataset, f)
    gaussian_noise_morphology_dataset, gaussian_noise_temporal_dataset = gaussian_noise_creation(real_data, real_landmarks)
    with open(morphology_path / "gaussian_noise_dataset.pkl", "wb") as f:
        pickle.dump(gaussian_noise_morphology_dataset, f)
    with open(temporal_path / "gaussian_noise_dataset.pkl", "wb") as f:
        pickle.dump(gaussian_noise_temporal_dataset, f)
    baseline_drift_morphology_dataset, baseline_drift_temporal_dataset = baseline_drift_creation(real_data, real_landmarks)
    with open(morphology_path / "baseline_drift_dataset.pkl", "wb") as f:
        pickle.dump(baseline_drift_morphology_dataset, f)
    with open(temporal_path / "baseline_drift_dataset.pkl", "wb") as f:
        pickle.dump(baseline_drift_temporal_dataset, f)
    spurious_transient_morphology_dataset, spurious_transient_temporal_dataset = spurious_transient_creation(real_data, real_landmarks)
    with open(morphology_path / "spurious_transient_dataset.pkl", "wb") as f:
        pickle.dump(spurious_transient_morphology_dataset, f)
    with open(temporal_path / "spurious_transient_dataset.pkl", "wb") as f:
        pickle.dump(spurious_transient_temporal_dataset, f)

    # Distributional Flawed Dataset Creation
    mode_collapse_vary_modes_morphology_dataset, mode_collapse_vary_modes_temporal_dataset = mode_collapse_vary_modes_creation(real_data, real_landmarks)
    with open(morphology_path / "mode_collapse_vary_modes_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_modes_morphology_dataset, f)
    with open(temporal_path / "mode_collapse_vary_modes_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_modes_temporal_dataset, f)
    mode_collapse_vary_spike_ratio_morphology_dataset, mode_collapse_vary_spike_ratio_temporal_dataset = mode_collapse_vary_spike_ratio_creation(real_data, real_landmarks)
    with open(morphology_path / "mode_collapse_vary_spike_ratio_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_spike_ratio_morphology_dataset, f)
    with open(temporal_path / "mode_collapse_vary_spike_ratio_dataset.pkl", "wb") as f:
        pickle.dump(mode_collapse_vary_spike_ratio_temporal_dataset, f)

    # Temporal Flawed Dataset Creation
    phase_shift_morphology_dataset, phase_shift_temporal_dataset = phase_shift_creation(real_data, real_landmarks)
    with open(morphology_path / "phase_shift_dataset.pkl", "wb") as f:
        pickle.dump(phase_shift_morphology_dataset, f)
    with open(temporal_path / "phase_shift_dataset.pkl", "wb") as f:
        pickle.dump(phase_shift_temporal_dataset, f)
    time_distortion_morphology_dataset, time_distortion_temporal_dataset = time_distortion_creation(real_data, real_landmarks)
    with open(morphology_path / "time_distortion_dataset.pkl", "wb") as f:
        pickle.dump(time_distortion_morphology_dataset, f)
    with open(temporal_path / "time_distortion_dataset.pkl", "wb") as f:
        pickle.dump(time_distortion_temporal_dataset, f)
    phase_jitter_morphology_dataset, phase_jitter_temporal_dataset = phase_jitter_creation(real_data, real_landmarks)
    with open(morphology_path / "phase_jitter_dataset.pkl", "wb") as f:
        pickle.dump(phase_jitter_morphology_dataset, f)
    with open(temporal_path / "phase_jitter_dataset.pkl", "wb") as f:
        pickle.dump(phase_jitter_temporal_dataset, f)
    loss_of_autocorrelation_morphology_dataset, loss_of_autocorrelation_temporal_dataset = loss_of_autocorrelation_creation(real_data, real_landmarks)
    with open(morphology_path / "loss_of_autocorrelation_dataset.pkl", "wb") as f:
        pickle.dump(loss_of_autocorrelation_morphology_dataset, f)
    with open(temporal_path / "loss_of_autocorrelation_dataset.pkl", "wb") as f:
        pickle.dump(loss_of_autocorrelation_temporal_dataset, f)

    # Privacy Flawed Dataset Creation
    if real_fd.data_matrix.shape[0] >= substitute_fd.data_matrix.shape[0]:
        common_grid = real_fd.grid_points[0]
    else:
        common_grid = substitute_fd.grid_points[0]
    real_fd_data = real_fd(common_grid)
    substitute_fd_data = substitute_fd(common_grid)
    holdout_train, holdout_test = train_test_split(substitute_fd_data, test_size=0.3, random_state=42)
    with open(privacy_path / "real_data.pkl", "wb") as f:
        pickle.dump(real_fd_data, f)
    with open(privacy_path / "holdout_train.pkl", "wb") as f:
        pickle.dump(holdout_train, f)
    with open(privacy_path / "holdout_test.pkl", "wb") as f:
        pickle.dump(holdout_test, f)

    real_fd = FDataGrid(data_matrix=real_fd_data, grid_points=common_grid)
    n_sample, n_timepoints, n_channel = real_fd.data_matrix.shape
    n_basis = int(n_timepoints / 2)
    lambda_ = basis_smoothing_hyperparameter_tuning(real_fd, n_basis, (0,1))
    real_fd_smooth, _, _, _ = basis_smoothing_with_lambda(real_fd, lambda_, n_basis, (0,1))
    real_mean, real_components, real_scores, real_var_ratio, real_fpca_ = fpca_with_param(real_fd_smooth, 10)

    holdout_train_fd = FDataGrid(data_matrix=holdout_train, grid_points=common_grid)
    lambda_ = basis_smoothing_hyperparameter_tuning(holdout_train_fd, n_basis, (0,1))
    holdout_train_fd_smooth, _, _, _ = basis_smoothing_with_lambda(holdout_train_fd, lambda_, n_basis, (0,1))
    holdout_scores = real_fpca_.transform(holdout_train_fd_smooth)

    privacy_leaking_dataset = privacy_leaking_creation(real_components.data_matrix.squeeze(), real_scores, real_var_ratio)
    with open(privacy_path / "real_components.pkl", "wb") as f:
        pickle.dump(real_components.data_matrix.squeeze(), f)
    with open(privacy_path / "holdout_scores.pkl", "wb") as f:
        pickle.dump(holdout_scores, f)
    with open(privacy_path / "privacy_leaking_dataset.pkl", "wb") as f:
        pickle.dump(privacy_leaking_dataset, f)

    query_size = 500
    query_idx = np.random.choice(holdout_test.shape[0], size=query_size, replace=False)
    query_data = np.vstack([real_fd_data[query_idx], holdout_test[query_idx]])
    query_label = np.vstack([np.ones(query_size), np.zeros(query_size)])
    
    query_fd = FDataGrid(data_matrix=query_data, grid_points=common_grid)
    lambda_ = basis_smoothing_hyperparameter_tuning(query_fd, n_basis, (0,1))
    query_fd_smooth, _, _, _ = basis_smoothing_with_lambda(query_fd, lambda_, n_basis, (0,1))
    query_scores = real_fpca_.transform(query_fd_smooth)
    
    with open(privacy_path / "query_data.pkl", "wb") as f:
        pickle.dump(query_data, f)
    with open(privacy_path / "query_label.pkl", "wb") as f:
        pickle.dump(query_label, f)
    with open(privacy_path / "query_scores.pkl", "wb") as f:
        pickle.dump(query_scores, f)
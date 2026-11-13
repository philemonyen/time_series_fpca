import ast
import pickle
import wfdb
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from skfda.representation import FDataGrid

#### ---- Dataset Source ----  ####
# https://physionet.org/content/ptb-xl/1.0.3/

#### ---- Global Parameters ---- ####
path = "data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
sampling_rate=100
diagnostics = np.array(['NORM', 'MI', 'STTC', 'CD', 'HYP'])

# ptbxl_database.csv columns saved alongside each ECG, in this order
demographic_columns = ['ecg_id', 'patient_id', 'age', 'sex', 'height', 'weight', 'strat_fold']

def get_diagnostics():
    return diagnostics

def get_sr():
    return sampling_rate

def get_demographic_columns():
    return demographic_columns


#### ---- Load PTB-XL Dataset ---- ####
_diagnostic_statements = None

def _get_diagnostic_statements():
    """scp_statements.csv restricted to diagnostic codes, read once per process."""
    global _diagnostic_statements
    if _diagnostic_statements is None:
        agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
        _diagnostic_statements = agg_df[agg_df.diagnostic == 1]
    return _diagnostic_statements

def aggregate_diagnostic(y_dic):
    agg_df = _get_diagnostic_statements()
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def get_data():
    """
    Load PTB-XL dataset into .npy files, with the demographics of each record
    written to a parallel .pkl file.

    For every diagnostic/sampling rate pair this writes
    `{diagnostic}_{sampling_rate}.npy`, shaped (n_records, n_leads, n_samples),
    and `{diagnostic}_{sampling_rate}_demographics.pkl`, a list of n_records
    lists holding `demographic_columns` in order. Row i of the pickle is the
    record in row i of the array.
    """
    save_path = "data"
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    noise_cols = ['baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems', 'extra_beats', 'pacemaker']
    Y = Y[Y[noise_cols].isna().all(axis=1)]

    # Work with normal ECG now
    for sampling_rate in [100, 500]:
        filename_col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
        for diagnostic in diagnostics:
            # keep records whose only superclass is this diagnostic, in ecg_id order
            records = Y[Y.diagnostic_superclass.apply(lambda sc, d=diagnostic: sc == [d])]

            data = [wfdb.rdsamp(path+f) for f in records[filename_col]]
            data = np.array([signal for signal, meta in data])
            data = data.transpose(0, 2, 1) # (n_records, n_leads, n_samples)

            np.save(save_path + f"/{diagnostic}_{sampling_rate}.npy", data)

            demographics = records.reset_index()[demographic_columns].to_numpy(dtype=object).tolist()
            with open(save_path + f"/{diagnostic}_{sampling_rate}_demographics.pkl", "wb") as f:
                pickle.dump(demographics, f)

#### ---- Load Dataset ---- ####
def load_dataset(diagnostic, sampling_rate, lead=None):
    data = np.load(f"data/{diagnostic}_{sampling_rate}.npy")
    if lead is not None:
        data = data[:, lead, :]
    return data

def load_demographics(diagnostic, sampling_rate):
    """Demographics of `load_dataset(diagnostic, sampling_rate)`, row for row."""
    with open(f"data/{diagnostic}_{sampling_rate}_demographics.pkl", "rb") as f:
        return pickle.load(f)

def load_synthetic_dataset(diagnostic, lead):
    data = np.load("data/synthetic_final.npy")
    label = np.load("data/synthetic_final_labels.npy")

    class_index = np.where(diagnostics == diagnostic)[0][0]
    mask = (label[:, class_index] == 1)

    filtered = data[mask][:, lead, :]

    return filtered

#### --- ECG Preprocess ---- ####
# Helper functions to find closest features
def closest_before(target, arr):
    valid = arr[arr < target]
    return valid[-1] if len(valid) > 0 else None

def closest_after(target, arr):
    valid = arr[arr > target]
    return valid[0] if len(valid) > 0 else None

def get_landmarks(signals, sr):
    max_pr_samples = int(0.30 * sr)  # 300ms max PR interval
    max_rt_samples = int(0.45 * sr)  # 450ms max RT interval
    landmarks = []
    
    for signal in signals:
        # 1. Feature Extraction
        # nk.ecg_process returns 1 at peak locations, 0 otherwise. np.where gets the indices.
        df, _ = nk.ecg_process(signal, sampling_rate=sr)
        p_onsets = np.where(df['ECG_P_Onsets'] == 1)[0]
        r_peaks = np.where(df['ECG_R_Peaks'] == 1)[0]
        t_offsets = np.where(df['ECG_T_Offsets'] == 1)[0]
        
        if len(r_peaks) == 0:
            continue # Skip completely invalid signals

        # Calculate medians for this specific signal for imputation
        pr_dists = [r - closest_before(r, p_onsets) for r in r_peaks if closest_before(r, p_onsets) is not None]
        rt_dists = [closest_after(r, t_offsets) - r for r in r_peaks if closest_after(r, t_offsets) is not None]
        
        med_pr = int(np.median(pr_dists)) if pr_dists else int(0.16 * sr) # Default 160ms
        med_rt = int(np.median(rt_dists)) if rt_dists else int(0.30 * sr) # Default 300ms
        
        # 2. R-Centric Triplet Matching & Imputation
        n_samples = len(signal)
        valid_beats = []
        for idx, r in enumerate(r_peaks):
            p = closest_before(r, p_onsets)
            t = closest_after(r, t_offsets)
            
            is_first = (idx == 0)
            is_last = (idx == len(r_peaks) - 1)
            
            # Check and impute P-onset
            if p is None or (r - p) > max_pr_samples:
                if is_first: continue # Drop if it's the boundary beat
                p = r - med_pr
                
            # Check and impute T-offset
            if t is None or (t - r) > max_rt_samples:
                if is_last: continue # Drop if it's the boundary beat
                t = r + med_rt

            p = int(np.clip(p, 0, n_samples - 1))
            r = int(np.clip(r, 0, n_samples - 1))
            t = int(np.clip(t, 0, n_samples - 1))
            if not (p <= r <= t):
                continue
                
            valid_beats.append((p, r, t))
            
        if not valid_beats:
            continue
        landmarks.append(valid_beats)

    if not landmarks:
        return np.empty((0, 0, 3), dtype=int)

    max_beats = max(len(beats) for beats in landmarks)
    padded = np.full((len(landmarks), max_beats, 3), -1, dtype=int)
    for i, beats in enumerate(landmarks):
        padded[i, :len(beats)] = np.asarray(beats, dtype=int)
    return padded

def align_ecg(signals, landmarks, points_per_beat=100, max_beats=None):
    """
    Preprocesses ECG signals by anchoring P-onset, R-peak, and T-offset to a continuous
    phase domain, imputing missing anchors, and padding to a global maximum beat count.
    
    Parameters:
    - signals: list of 1D numpy arrays (the raw ECG signals)
    - landmarks: P/R/T indices on the original axis, shape (n_signals, max_beats, 3)
    - points_per_beat: int, resolution of the resampled phase grid per cardiac cycle
    - max_beats: optional fixed beat budget for the shared phase grid. When set, every
      curve is resampled onto ``max_beats * points_per_beat`` points so later batches
      can match an earlier alignment (e.g. FPCA fit). Signals with more beats are
      truncated; signals with fewer beats are zero-padded as before.
    """
    processed_data = []
    beat_counts = []
    
    for signal, landmark in zip(signals, landmarks):
        signal = np.asarray(signal, dtype=np.float64).reshape(-1)
        n_samples = len(signal)
        landmark = np.asarray(landmark, dtype=int)
        landmark = landmark[landmark[:, 1] >= 0]
        if len(landmark) == 0:
            continue
        if max_beats is not None and len(landmark) > max_beats:
            landmark = landmark[:max_beats]

        # Clamp P/R/T into the signal and keep only ordered beats.
        cleaned = []
        for p, r, t in landmark:
            p = int(np.clip(p, 0, n_samples - 1))
            r = int(np.clip(r, 0, n_samples - 1))
            t = int(np.clip(t, 0, n_samples - 1))
            if p <= r <= t:
                cleaned.append((p, r, t))
        if not cleaned:
            continue
        landmark = np.asarray(cleaned, dtype=int)

        # 3. Phase Projection Setup
        start_idx = int(landmark[0][0])
        end_idx = int(landmark[-1][2])
        if end_idx < start_idx:
            continue
        trimmed_signal = signal[start_idx:end_idx + 1]
        if len(trimmed_signal) == 0:
            continue
        
        known_indices = []
        known_phases = []
        
        for i, (p, r, t) in enumerate(landmark):
            # Shift indices relative to the trimmed start
            p_rel, r_rel, t_rel = p - start_idx, r - start_idx, t - start_idx
            
            known_indices.extend([p_rel, r_rel, t_rel])
            # Phase mapping: P -> 0, R -> pi, T -> 2pi
            known_phases.extend([i * 2*np.pi, i * 2*np.pi + np.pi, i * 2*np.pi + 2*np.pi])

        known_indices = np.asarray(known_indices, dtype=np.float64)
        known_phases = np.asarray(known_phases, dtype=np.float64)
        order = np.argsort(known_indices)
        known_indices = known_indices[order]
        known_phases = known_phases[order]
        # Drop duplicate sample anchors so np.interp gets strictly increasing xp.
        keep = np.ones(len(known_indices), dtype=bool)
        keep[1:] = np.diff(known_indices) > 0
        known_indices = known_indices[keep]
        known_phases = known_phases[keep]
        if len(known_indices) < 2:
            continue
            
        # 4. Create continuous phase axis for the trimmed signal
        all_indices = np.arange(len(trimmed_signal))
        # Linearly interpolate the phase for every point in the sequence
        continuous_phase = np.interp(all_indices, known_indices, known_phases)
        
        processed_data.append((continuous_phase, trimmed_signal))
        beat_counts.append(len(landmark))
        
    # 5. Global Shared Grid Alignment with Zero Padding
    if not beat_counts:
        raise ValueError("align_ecg received no signals with valid landmarks.")
    if max_beats is None:
        max_beats = max(beat_counts)
    total_points = max_beats * points_per_beat
    
    # Global phase grid from 0 to max_beats * 2pi
    common_phase_grid = np.linspace(0, max_beats * 2 * np.pi, total_points)
    
    resampled_data = []
    for continuous_phase, trimmed_signal in processed_data:
        if len(continuous_phase) == 0 or len(trimmed_signal) == 0:
            resampled_data.append(np.zeros(total_points, dtype=np.float64))
            continue
        # Interpolate the signal onto the common grid.
        # `right=0.0` automatically pads the signal with 0 (isoelectric line)
        # for phase values that exceed this specific signal's duration.
        resampled = np.interp(common_phase_grid, continuous_phase, trimmed_signal, left=0.0, right=0.0)
        resampled_data.append(resampled)
        
    # Convert to FDataGrid
    # The grid points are scaled between 0 and 1 representing the entire normalized timeline
    normalized_grid = np.linspace(0, 1, total_points)
    fd = FDataGrid(data_matrix=resampled_data, grid_points=normalized_grid)
    
    return fd


def reverse_align_ecg(processed_signals, landmarks, points_per_beat=100, original_length=1000):
    """
    Invert ``align_ecg``: map phase-aligned curves back onto the original time axis.

    Each heartbeat on the aligned grid is warped back to its original P/R/T sample
    indices. Samples before the first P-onset, after the last T-offset, and in any
    gaps between consecutive beats are filled by linear interpolation (zeros outside
    the landmark span when no edge sample is available).

    Parameters
    ----------
    processed_signals : array-like or FDataGrid
        Aligned curves of shape ``(n_signals, n_aligned)`` (or an ``FDataGrid`` whose
        ``data_matrix`` has that leading shape). ``n_aligned`` must equal
        ``max_beats * points_per_beat`` from the forward alignment.
    landmarks : array-like, shape ``(n_signals, max_beats, 3)``
        Same P-onset / R-peak / T-offset indices (on the original axis) that were
        passed to ``align_ecg``. Invalid beats are marked with R < 0.
    points_per_beat : int
        Must match the value used in ``align_ecg``.
    original_length : int
        Length of the restored traces (1000 for PTB-XL at 100 Hz).

    Returns
    -------
    restored : ndarray, shape ``(n_signals, original_length)``
    """
    if isinstance(processed_signals, FDataGrid):
        curves = np.asarray(processed_signals.data_matrix, dtype=np.float64)
    else:
        curves = np.asarray(processed_signals, dtype=np.float64)
    if curves.ndim == 3:
        curves = curves.reshape(curves.shape[0], -1)
    elif curves.ndim == 1:
        curves = curves[None, :]
    if curves.ndim != 2:
        raise ValueError(f"processed_signals must be 1D or 2D, got shape {curves.shape}")

    landmarks = np.asarray(landmarks)
    if landmarks.ndim != 3 or landmarks.shape[-1] != 3:
        raise ValueError(
            f"landmarks must have shape (n_signals, max_beats, 3), got {landmarks.shape}"
        )
    if len(curves) != len(landmarks):
        raise ValueError(
            f"processed_signals ({len(curves)}) and landmarks ({len(landmarks)}) length differ"
        )

    n_aligned = curves.shape[1]
    if n_aligned % points_per_beat != 0:
        raise ValueError(
            f"aligned length {n_aligned} is not divisible by points_per_beat={points_per_beat}"
        )
    max_beats = n_aligned // points_per_beat
    common_phase_grid = np.linspace(0.0, max_beats * 2.0 * np.pi, n_aligned)

    restored = np.zeros((len(curves), original_length), dtype=np.float64)

    for i, (curve, landmark) in enumerate(zip(curves, landmarks)):
        beats = landmark[landmark[:, 1] >= 0]
        if len(beats) == 0:
            continue

        # Phase -> original time using the same P/R/T anchors as align_ecg.
        known_phases = []
        known_times = []
        for b, (p, r, t) in enumerate(beats):
            known_phases.extend([b * 2.0 * np.pi, b * 2.0 * np.pi + np.pi, b * 2.0 * np.pi + 2.0 * np.pi])
            known_times.extend([float(p), float(r), float(t)])
        known_phases = np.asarray(known_phases, dtype=np.float64)
        known_times = np.asarray(known_times, dtype=np.float64)
        # Consecutive T / next-P share a phase; nudge so interp is well-defined.
        for k in range(1, len(known_phases)):
            if known_phases[k] <= known_phases[k - 1]:
                known_phases[k] = known_phases[k - 1] + 1e-12

        n_beats = len(beats)
        content = common_phase_grid <= (n_beats * 2.0 * np.pi + 1e-12)
        phases = common_phase_grid[content]
        values = curve[content]
        orig_times = np.interp(phases, known_phases, known_times)

        order = np.argsort(orig_times, kind="mergesort")
        orig_times = orig_times[order]
        values = values[order]
        # Collapse samples that land on the same original index.
        uniq_t, inv = np.unique(np.round(orig_times, decimals=8), return_inverse=True)
        uniq_v = np.zeros(len(uniq_t), dtype=np.float64)
        counts = np.zeros(len(uniq_t), dtype=np.float64)
        np.add.at(uniq_v, inv, values)
        np.add.at(counts, inv, 1.0)
        uniq_v /= counts

        start_idx = int(beats[0, 0])
        end_idx = int(beats[-1, 2])
        start_idx = max(0, min(start_idx, original_length - 1))
        end_idx = max(0, min(end_idx, original_length - 1))
        if end_idx < start_idx:
            continue

        target = np.arange(start_idx, end_idx + 1, dtype=np.float64)
        restored[i, start_idx : end_idx + 1] = np.interp(target, uniq_t, uniq_v)

        # Gaps between T of beat b and P of beat b+1: re-interpolate on the
        # original axis so the join is smooth even when the forward map collapsed them.
        for b in range(n_beats - 1):
            left = int(beats[b, 2])
            right = int(beats[b + 1, 0])
            if right - left <= 1:
                continue
            left = max(0, min(left, original_length - 1))
            right = max(0, min(right, original_length - 1))
            gap = np.arange(left, right + 1)
            restored[i, left : right + 1] = np.interp(
                gap.astype(np.float64),
                np.array([float(left), float(right)]),
                np.array([restored[i, left], restored[i, right]]),
            )

    return restored


def extract_ecg_sliding_windows(signals, landmarks, window_beats=8, points_per_window=1000):
    """
    Extracts fixed-beat sliding windows from ECG signals and pools them globally.
    Normalizes each extracted window to a domain of t in [0, 1] and calculates 
    the relative landmark locations for subsequent temporal registration.
    
    Parameters:
    - signals: list of 1D numpy arrays (raw ECG signals)
    - sr: int, sampling rate
    - window_beats: int, the fixed number of R-peaks (N) per sliding window
    - points_per_window: int, the resolution of the shared spatial grid for FDataGrid
    
    Returns:
    - fd: FDataGrid containing all pooled, time-normalized windows
    - global_landmarks: ndarray of shape (total_windows, window_beats) containing 
                        the relative locations of the R-peaks in [0, 1]
    """
    pooled_data = []
    global_landmarks = []
    
    # Common domain for the FDataGrid
    normalized_grid = np.linspace(0, 1, points_per_window)
    
    for signal, landmark in zip(signals, landmarks):
        landmark = landmark[landmark[:, 1] >= 0]
        if len(landmark) == 0:
            continue

        r_peaks = np.array(landmark)[:, 1]
        r_peaks.sort()
        if 0 in np.diff(r_peaks): continue
        if len(r_peaks) < window_beats: continue
            
        # 2. Sliding Window Extraction
        # Slide across the R-peaks array: e.g., for N=6, indices 0:6, 1:7, 2:8...
        for i in range(len(r_peaks) - window_beats + 1):
            window_r_peaks = r_peaks[i : i + window_beats]
            
            start_idx = window_r_peaks[0]
            end_idx = window_r_peaks[-1]
            
            # Extract the raw signal spanning this specific N-beat epoch
            trimmed_signal = signal[start_idx : end_idx + 1]
            
            # 3. Domain Normalization (t in [0, 1])
            # Calculate the relative positions of the landmarks within this window
            window_length = end_idx - start_idx
            relative_landmarks = (window_r_peaks - start_idx) / window_length
            
            # Create the original time grid for this trimmed signal (0 to 1)
            original_grid = np.linspace(0, 1, len(trimmed_signal))
            
            # Interpolate the raw signal onto the shared global grid size
            resampled_signal = np.interp(normalized_grid, original_grid, trimmed_signal)
            
            pooled_data.append(resampled_signal)
            global_landmarks.append(relative_landmarks[1:-1]) # keep only the internal landmarks
            
    # 4. Global Pooling
    # Convert pooled arrays into the skfda compatible formats
    fd = FDataGrid(data_matrix=pooled_data, grid_points=normalized_grid)
    global_landmarks = np.array(global_landmarks)
    
    return fd, global_landmarks

if __name__ == "__main__":
    # diagnostic = "NORM"
    # lead = 1
    # sr = get_sr()

    # # Get Real Data
    # real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)
    # landmarks = get_landmarks(real_all, sr)
    # fd = align_ecg(real_all, landmarks)
    # # fd, global_landmarks = extract_ecg_sliding_windows(real_all, landmarks)
    # print(fd.data_matrix.shape) # Morphology: (6804, 2200, 1), Temporal: (29891, 1000, 1)
    # fd.plot()
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude (mV)")
    # plt.title("Processed ECG")
    # plt.show()
    get_data()
    

    
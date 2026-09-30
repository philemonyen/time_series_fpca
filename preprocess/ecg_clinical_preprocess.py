import pickle
from pathlib import Path
import numpy as np
import neurokit2 as nk
from preprocess.ptbxl_preprocess import load_dataset 

def closest_before(target, arr):
    valid = arr[arr < target]
    return valid[-1] if len(valid) > 0 else None

def closest_after(target, arr):
    valid = arr[arr > target]
    return valid[0] if len(valid) > 0 else None

def _event_indices(df, column):
    if column not in df.columns:
        return np.array([], dtype=int)
    return np.where(df[column].to_numpy() == 1)[0]

def _median_samples(distances, default):
    if len(distances) == 0:
        return default
    return max(1, int(np.median(distances)))

def _pad_feature(rows):
    rows = [np.asarray(row, dtype=float) for row in rows]
    max_beats = max((row.size for row in rows), default=0)
    padded = np.full((len(rows), max_beats), np.nan, dtype=float)
    for i, row in enumerate(rows):
        if row.size:
            padded[i, :row.size] = row
    return padded

def _empty_features():
    return tuple(np.array([], dtype=float) for _ in range(6))

def _r_peaks_from_landmarks(cleaned, marks, min_separation):
    """Landmarks whose absolute amplitude dominates the surrounding minimum RR."""
    amplitudes = np.abs(cleaned[marks])
    peaks = []
    for i, idx in enumerate(marks):
        near = np.abs(marks - idx) <= min_separation
        if amplitudes[i] < amplitudes[near].max():
            continue
        if peaks and int(idx) - peaks[-1] < min_separation:
            continue
        peaks.append(int(idx))
    return np.asarray(peaks, dtype=int)

def get_pqrst(signals, sr):
    """Beat-wise (P, Q, R, S, T) landmarks.

    Each detected R peak anchors one tuple
    ``(P onset, QRS onset, R peak, QRS offset, T offset)``. A missing or
    implausibly timed mark is -1. Rows are padded with -1 to a common beat
    count, so the result has shape ``(n_signals, max_beats, 5)``.
    """
    max_pr = int(0.30 * sr)
    max_qon = int(0.12 * sr)
    max_qoff = int(0.12 * sr)
    max_rt = int(0.45 * sr)

    landmarks = []
    for signal in signals:
        try:
            df, _ = nk.ecg_process(np.asarray(signal, dtype=float), sampling_rate=sr)
        except Exception:
            landmarks.append([])
            continue

        p_onsets = _event_indices(df, "ECG_P_Onsets")
        qrs_onsets = _event_indices(df, "ECG_R_Onsets")
        r_peaks = _event_indices(df, "ECG_R_Peaks")
        qrs_offsets = _event_indices(df, "ECG_R_Offsets")
        t_offsets = _event_indices(df, "ECG_T_Offsets")

        beats = []
        n_peaks = len(r_peaks)
        for idx, r in enumerate(r_peaks):
            q = closest_before(r, qrs_onsets)
            if q is None or not (0 < r - q <= max_qon):
                q = -1
            elif idx > 0 and q <= r_peaks[idx - 1]:
                q = -1

            if q == -1:
                p = -1
            else:
                p = closest_before(q, p_onsets)
                if p is None or not (0 < q - p <= max_pr):
                    p = -1
                elif idx > 0 and p <= r_peaks[idx - 1]:
                    p = -1

            s = closest_after(r, qrs_offsets)
            if s is None or not (0 < s - r <= max_qoff):
                s = -1
            elif idx + 1 < n_peaks and s >= r_peaks[idx + 1]:
                s = -1

            t = closest_after(r, t_offsets)
            if t is None or not (0 < t - r <= max_rt):
                t = -1
            elif s != -1 and t <= s:
                t = -1
            elif idx + 1 < n_peaks and t >= r_peaks[idx + 1]:
                t = -1

            beats.append((int(p), int(q), int(r), int(s), int(t)))
        landmarks.append(beats)

    max_beats = max((len(beats) for beats in landmarks), default=0)
    padded = np.full((len(landmarks), max_beats, 5), -1, dtype=int)
    for i, beats in enumerate(landmarks):
        if beats:
            padded[i, :len(beats)] = np.asarray(beats, dtype=int)
    return padded

def get_clinical_features(signals, sr):
    """
    Beat-wise clinical features for a collection of single-lead ECGs.

    Each detected R peak anchors one beat. Missing or implausibly timed
    P-onset, QRS-onset, QRS-offset, and T-offset marks are imputed from that
    recording's median timing. Recordings with no usable measurements fall
    back to normal-adult defaults. As in the landmark imputation, the first
    beat is dropped when its P onset or QRS onset cannot be imputed, and the
    last beat is dropped when its QRS offset or T offset cannot be imputed.
    The first beat has no preceding RR interval, so that entry uses the
    recording's median RR.

    Parameters
    ----------
    signals : sequence of 1D arrays
        One single-lead ECG per recording, in signal units (mV for PTB-XL).
    sr : int
        Sampling rate in Hz.

    Returns
    -------
    pr, rr, qrs, qt, r_amp, qrs_area : ndarray
        Six arrays of shape (n_signals, max_beats). Row i corresponds to
        signals[i]. pr, rr, qrs, and qt are in seconds:
        PR is P onset to QRS onset, RR is the preceding R-to-R gap,
        QRS is QRS onset to QRS offset, and QT is QRS onset to T offset.
        r_amp is the cleaned-signal voltage at the R peak.
        qrs_area is the absolute area, in voltage * seconds, of the cleaned
        signal from QRS onset to QRS offset after removing the linear
        baseline between those samples.
        Beats beyond a recording's last accepted beat are NaN.
    """
    max_pr = int(0.30 * sr)    
    max_qon = int(0.12 * sr)   
    max_qoff = int(0.12 * sr)  
    max_rt = int(0.45 * sr)    
    min_rr = int(0.30 * sr)    
    max_rr = int(2.00 * sr)    

    default_pr = int(0.16 * sr)     
    default_qon = int(0.04 * sr)    
    default_qoff = int(0.05 * sr)   
    default_rt = int(0.30 * sr)     
    default_rr = int(0.80 * sr)     

    features = []
    for signal in signals:
        try:
            df, _ = nk.ecg_process(np.asarray(signal, dtype=float), sampling_rate=sr)
        except Exception:
            features.append(_empty_features())
            continue

        cleaned = df["ECG_Clean"].to_numpy(dtype=float)
        p_onsets = _event_indices(df, "ECG_P_Onsets")
        qrs_onsets = _event_indices(df, "ECG_R_Onsets")
        r_peaks = _event_indices(df, "ECG_R_Peaks")
        qrs_offsets = _event_indices(df, "ECG_R_Offsets")
        t_offsets = _event_indices(df, "ECG_T_Offsets")

        if len(r_peaks) == 0:
            features.append(_empty_features())
            continue

        pr_dists, qon_dists, qoff_dists, rt_dists = [], [], [], []
        for r in r_peaks:
            q_on = closest_before(r, qrs_onsets)
            if q_on is not None and 0 < r - q_on <= max_qon:
                qon_dists.append(r - q_on)
                p = closest_before(q_on, p_onsets)
                if p is not None and 0 < q_on - p <= max_pr:
                    pr_dists.append(q_on - p)
            q_off = closest_after(r, qrs_offsets)
            if q_off is not None and 0 < q_off - r <= max_qoff:
                qoff_dists.append(q_off - r)
            t = closest_after(r, t_offsets)
            if t is not None and 0 < t - r <= max_rt:
                rt_dists.append(t - r)

        rr_gaps = np.diff(r_peaks)
        plausible_rr = rr_gaps[(rr_gaps >= min_rr) & (rr_gaps <= max_rr)]
        med_pr = _median_samples(pr_dists, default_pr)
        med_qon = _median_samples(qon_dists, default_qon)
        med_qoff = _median_samples(qoff_dists, default_qoff)
        med_rt = _median_samples(rt_dists, default_rt)
        med_rr = _median_samples(plausible_rr, default_rr)

        pr, rr, qrs, qt, r_amp, qrs_area = [], [], [], [], [], []
        n_samples = len(cleaned)
        
        for idx, r in enumerate(r_peaks):
            is_first = idx == 0
            is_last = idx == len(r_peaks) - 1

            q_on = closest_before(r, qrs_onsets)
            if q_on is None or not (0 < r - q_on <= max_qon):
                if is_first:
                    continue
                q_on = r - med_qon

            p = closest_before(q_on, p_onsets)
            if p is None or not (0 < q_on - p <= max_pr):
                if is_first:
                    continue
                p = q_on - med_pr

            q_off = closest_after(r, qrs_offsets)
            if q_off is None or not (0 < q_off - r <= max_qoff):
                if is_last:
                    continue
                q_off = r + med_qoff

            t = closest_after(r, t_offsets)
            if t is None or not (0 < t - r <= max_rt):
                if is_last:
                    continue
                t = r + med_rt

            p, q_on, r_i, q_off, t = int(p), int(q_on), int(r), int(q_off), int(t)
            if not (0 <= p < q_on < r_i < q_off < t < n_samples):
                continue
            if not is_first and p <= r_peaks[idx - 1]:
                continue
            if not is_last and t >= r_peaks[idx + 1]:
                continue

            if is_first:
                rr_i = med_rr
            else:
                rr_i = r_i - int(r_peaks[idx - 1])
                if not (min_rr <= rr_i <= max_rr):
                    rr_i = med_rr

            pr.append((q_on - p) / sr)
            rr.append(rr_i / sr)
            qrs.append((q_off - q_on) / sr)
            qt.append((t - q_on) / sr)

            # ==========================================
            # NEW: R Peak Amplitude and QRS Area Extraction
            # ==========================================
            
            # 1. R-Peak Voltage
            r_amp.append(cleaned[r_i])
            
            # 2. QRS Area Calculation
            # Slice the cleaned signal strictly from q_on to q_off (inclusive)
            qrs_segment = cleaned[q_on:q_off + 1]
            
            # Create a linear baseline connecting the start (Q_on) and end (Q_off) of the segment
            baseline = np.linspace(cleaned[q_on], cleaned[q_off], len(qrs_segment))
            
            # Remove the linear baseline to center the QRS complex
            adjusted_segment = qrs_segment - baseline
            
            # Calculate the absolute area using the Trapezoidal rule
            # Since dx = 1 / sr, the area is natively measured in voltage * seconds
            area = np.trapz(np.abs(adjusted_segment), dx=1/sr)
            qrs_area.append(area)

        if not pr:
            features.append(_empty_features())
            continue

        # Append ALL 6 computed arrays to features
        features.append((
            np.asarray(pr, dtype=float),
            np.asarray(rr, dtype=float),
            np.asarray(qrs, dtype=float),
            np.asarray(qt, dtype=float),
            np.asarray(r_amp, dtype=float),
            np.asarray(qrs_area, dtype=float)
        ))

    columns = list(zip(*features)) if features else [() for _ in range(6)]
    return tuple(_pad_feature(rows) for rows in columns)


def get_clinical_features_from_pqrst(signals, pqrst, sr):
    """
    Beat-wise clinical features from get_pqrst landmarks.

    Uses the same intervals, imputation, and amplitude measurements as
    get_clinical_features. Each recording is a (n_beats, 5) table of
    (P onset, QRS onset, R peak, QRS offset, T offset) indices, with -1 for
    missing marks. The R peak of a beat is the landmark with the largest
    absolute amplitude within the minimum RR. QRS onset and P onset are the
    preceding landmarks, and QRS offset and T offset are the following ones.
    Missing or implausibly timed marks are imputed from that recording's
    median timing, with the same first-beat and last-beat drops.

    Parameters
    ----------
    signals : sequence of 1D arrays
        One single-lead ECG per recording, in signal units (mV for PTB-XL).
    pqrst : array, shape (n_signals, max_beats, 5)
        Output of get_pqrst. Each beat is (P, Q, R, S, T), padded with -1.
    sr : int
        Sampling rate in Hz.

    Returns
    -------
    pr, rr, qrs, qt, r_amp, qrs_area : ndarray
        Six arrays of shape (n_signals, max_beats), matching
        get_clinical_features.
    """
    max_pr = int(0.30 * sr)
    max_qon = int(0.12 * sr)
    max_qoff = int(0.12 * sr)
    max_rt = int(0.45 * sr)
    min_rr = int(0.30 * sr)
    max_rr = int(2.00 * sr)

    default_pr = int(0.16 * sr)
    default_qon = int(0.04 * sr)
    default_qoff = int(0.05 * sr)
    default_rt = int(0.30 * sr)
    default_rr = int(0.80 * sr)

    features = []
    for signal, marks in zip(signals, pqrst):
        signal = np.asarray(signal, dtype=float)
        marks = np.asarray(marks, dtype=int).ravel()
        marks = np.unique(marks[(marks >= 0) & (marks < signal.size)])
        if marks.size == 0:
            features.append(_empty_features())
            continue

        try:
            cleaned = np.asarray(nk.ecg_clean(signal, sampling_rate=sr), dtype=float)
        except Exception:
            cleaned = signal

        r_peaks = _r_peaks_from_landmarks(cleaned, marks, min_rr)
        if r_peaks.size == 0:
            features.append(_empty_features())
            continue

        others = marks[~np.isin(marks, r_peaks)]
        pr_dists, qon_dists, qoff_dists, rt_dists = [], [], [], []
        for r in r_peaks:
            q_on = closest_before(r, others)
            if q_on is not None and 0 < r - q_on <= max_qon:
                qon_dists.append(r - q_on)
                p = closest_before(q_on, others)
                if p is not None and 0 < q_on - p <= max_pr:
                    pr_dists.append(q_on - p)
            q_off = closest_after(r, others)
            if q_off is not None and 0 < q_off - r <= max_qoff:
                qoff_dists.append(q_off - r)
                t = closest_after(q_off, others)
                if t is not None and 0 < t - r <= max_rt:
                    rt_dists.append(t - r)

        rr_gaps = np.diff(r_peaks)
        plausible_rr = rr_gaps[(rr_gaps >= min_rr) & (rr_gaps <= max_rr)]
        med_pr = _median_samples(pr_dists, default_pr)
        med_qon = _median_samples(qon_dists, default_qon)
        med_qoff = _median_samples(qoff_dists, default_qoff)
        med_rt = _median_samples(rt_dists, default_rt)
        med_rr = _median_samples(plausible_rr, default_rr)

        pr, rr, qrs, qt, r_amp, qrs_area = [], [], [], [], [], []
        n_samples = len(cleaned)
        for idx, r in enumerate(r_peaks):
            is_first = idx == 0
            is_last = idx == len(r_peaks) - 1

            q_on = closest_before(r, others)
            if q_on is None or not (0 < r - q_on <= max_qon):
                if is_first:
                    continue
                q_on = r - med_qon

            p = closest_before(q_on, others)
            if p is None or not (0 < q_on - p <= max_pr):
                if is_first:
                    continue
                p = q_on - med_pr

            q_off = closest_after(r, others)
            if q_off is None or not (0 < q_off - r <= max_qoff):
                if is_last:
                    continue
                q_off = r + med_qoff

            t = closest_after(q_off, others)
            if t is None or not (0 < t - r <= max_rt):
                if is_last:
                    continue
                t = r + med_rt

            p, q_on, r_i, q_off, t = int(p), int(q_on), int(r), int(q_off), int(t)
            if not (0 <= p < q_on < r_i < q_off < t < n_samples):
                continue
            if not is_first and p <= r_peaks[idx - 1]:
                continue
            if not is_last and t >= r_peaks[idx + 1]:
                continue

            if is_first:
                rr_i = med_rr
            else:
                rr_i = r_i - int(r_peaks[idx - 1])
                if not (min_rr <= rr_i <= max_rr):
                    rr_i = med_rr

            pr.append((q_on - p) / sr)
            rr.append(rr_i / sr)
            qrs.append((q_off - q_on) / sr)
            qt.append((t - q_on) / sr)
            r_amp.append(cleaned[r_i])

            qrs_segment = cleaned[q_on:q_off + 1]
            baseline = np.linspace(cleaned[q_on], cleaned[q_off], len(qrs_segment))
            adjusted_segment = qrs_segment - baseline
            qrs_area.append(np.trapz(np.abs(adjusted_segment), dx=1 / sr))

        if not pr:
            features.append(_empty_features())
            continue

        features.append((
            np.asarray(pr, dtype=float),
            np.asarray(rr, dtype=float),
            np.asarray(qrs, dtype=float),
            np.asarray(qt, dtype=float),
            np.asarray(r_amp, dtype=float),
            np.asarray(qrs_area, dtype=float),
        ))

    columns = list(zip(*features)) if features else [() for _ in range(6)]
    return np.array(tuple(_pad_feature(rows) for rows in columns))

if __name__ == "__main__":
    diagnostic = "NORM"
    lead = 1
    sr = 100

    path = Path("data/clinical/")
    path.mkdir(parents=True, exist_ok=True)

    # Get Real Data
    real_all = load_dataset(diagnostic=diagnostic, sampling_rate=sr, lead=lead)

    # Unpack all 6 variables returned from the function
    pr, rr, qrs, qt, r_amp, qrs_area = get_clinical_features(real_all, sr)
    
    with open(path / f"pr_{diagnostic}_{lead}.pkl", "wb") as f:
        pickle.dump(pr, f)
    with open(path / f"rr_{diagnostic}_{lead}.pkl", "wb") as f:
        pickle.dump(rr, f)
    with open(path / f"qrs_{diagnostic}_{lead}.pkl", "wb") as f:
        pickle.dump(qrs, f)
    with open(path / f"qt_{diagnostic}_{lead}.pkl", "wb") as f:
        pickle.dump(qt, f)
        
    # Serialize the newly added features
    with open(path / f"r_amp_{diagnostic}_{lead}.pkl", "wb") as f:
        pickle.dump(r_amp, f)
    with open(path / f"qrs_area_{diagnostic}_{lead}.pkl", "wb") as f:
        pickle.dump(qrs_area, f)
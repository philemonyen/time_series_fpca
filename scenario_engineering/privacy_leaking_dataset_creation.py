"""
Privacy-leaking dataset creation for validating time-series privacy attacks (MIA, Re-identification).

Data Flow & Splitting:
1. Load lead II of NORM_100.npy. Landmark-align ECGs.
2. Split dataset at patient level into "real" (member) and "holdout" (non-member) cohorts.
3. Holdout is split into Shadow Train and Shadow Val.
4. "Real" records are used to engineer Leaky Target datasets:
    - FPC Based Synthetic Strategy:
      - Fit FPCA on Holdout Train.
      - Transform real ECGs to FPC scores.
      - Perturb scores and reconstruct.
      - Leaky targets are created by replacing a percentage (leak ratio) of 
        the perturbed traces with the exact real traces.
5. Membership Inference (MIA) datasets are explicitly saved with labels.
"""
import json
import os
from pathlib import Path

os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from skfda.representation.grid import FDataGrid

from preprocess.ptbxl_preprocess import (
    align_ecg,
    reverse_align_ecg,
    get_landmarks,
    get_sr,
    load_dataset,
)
from transformation.fda.fpca import (
    basis_smoothing_hyperparameter_tuning,
    basis_smoothing_with_lambda,
    fpca_with_param,
    fpca_hyperparameter_tuning,
)

def get_leak_ratios():
    return [0.10, 0.15, 0.20, 0.25, 0.30]

def get_segment_leak_scale():
    return [1, 2, 3, 4, 5]

def get_flawed_size():
    return 3000

def get_fpca_n_components():
    return 10

def load_real(diagnostic, lead):
    ts = load_dataset(diagnostic=diagnostic, sampling_rate=get_sr(), lead=lead)
    return ts

def split_real_and_holdout(n_total, rng, n_real=3000, holdout_val_fraction=0.3):
    indices = rng.permutation(n_total)
    
    real_idx = np.sort(indices[:n_real])
    remaining = indices[n_real:]
    
    n_val = int(round(len(remaining) * holdout_val_fraction))
    holdout_test_idx = np.sort(remaining[:n_val])
    holdout_train_idx = np.sort(remaining[n_val:])

    return real_idx, holdout_train_idx, holdout_test_idx

def subset_mask(n, ratio, order):
    mask = np.zeros(n, dtype=bool)
    mask[order[:int(round(ratio * n))]] = True
    return mask

def landmarks_per_signal(signals, sr):
    kept, beats = [], []
    for i, signal in enumerate(signals):
        try:
            landmark = get_landmarks(signal[None], sr)
        except Exception:
            continue
        if len(landmark) > 0:
            kept.append(i)
            beats.append(landmark[0])
    if not beats:
        raise ValueError("No member ECG had valid landmarks.")
    padded = np.full((len(beats), max(len(b) for b in beats), 3), -1, dtype=int)
    for i, beat in enumerate(beats):
        padded[i, :len(beat)] = beat
    return padded, np.asarray(kept)

def _as_curves(data):
    data = np.asarray(data, dtype=np.float64)
    return data.reshape(data.shape[0], -1)

def align_dataset(ts, sr, points_per_beat=100):
    """Landmark-align ECGs and keep the landmarks needed to undo the warp later."""
    landmarks, kept = landmarks_per_signal(ts, sr)
    fd = align_ecg(ts[kept], landmarks, points_per_beat=points_per_beat)
    grid = np.asarray(fd.grid_points[0], dtype=np.float64)
    return _as_curves(fd.data_matrix), kept, grid, landmarks

def resample_curves(curves, source_grid, target_grid):
    curves = _as_curves(curves)
    if len(source_grid) == len(target_grid) and np.allclose(source_grid, target_grid):
        return curves
    return np.vstack([np.interp(target_grid, source_grid, row) for row in curves])

def fit_fpca_model(aligned_ts, aligned_grid, n_components, domain_range=(0, 1)):
    fd = FDataGrid(data_matrix=aligned_ts, grid_points=aligned_grid)
    n_basis = int(aligned_ts.shape[1] / 2)
    lambda_ = basis_smoothing_hyperparameter_tuning(fd, n_basis, domain_range)
    fd_smooth, _, _, _ = basis_smoothing_with_lambda(fd, lambda_, n_basis, domain_range)
    n_components = fpca_hyperparameter_tuning(fd_smooth)
    _, _, _, _, fpca_ = fpca_with_param(fd_smooth, n_components)
    grid = np.asarray(fpca_.components_.grid_points[0], dtype=np.float64)
    return fpca_, grid

def transform_and_reconstruct(ts_arr, aligned_grid, fpca_, fpca_grid, domain_range=(0, 1)):
    fd = FDataGrid(data_matrix=ts_arr, grid_points=aligned_grid)
    n_basis = int(ts_arr.shape[1] / 2)
    lambda_ = basis_smoothing_hyperparameter_tuning(fd, n_basis, domain_range)
    fd_smooth, _, _, _ = basis_smoothing_with_lambda(fd, lambda_, n_basis, domain_range)
    fd_smooth = FDataGrid(data_matrix=fd_smooth(fpca_grid), grid_points=fpca_grid)
    scores = np.asarray(fpca_.transform(fd_smooth), dtype=np.float64)
    
    components = np.asarray(fpca_.components_.data_matrix, dtype=np.float64).reshape(fpca_.n_components, -1)
    mean = np.asarray(fpca_.mean_.data_matrix, dtype=np.float64).reshape(-1)
    reconstructed = scores @ components + mean
    
    original = resample_curves(ts_arr, aligned_grid, fpca_grid)
    residuals = _as_curves(original) - _as_curves(reconstructed)
    return scores, residuals

def perturb_fpca_scores(scores, rng, noise_multiplier):
    std = np.std(scores, axis=0, ddof=1)
    std = np.where(~np.isfinite(std) | (std < 1e-8), 1.0, std)
    return scores + rng.normal(0.0, noise_multiplier * std, size=scores.shape)

def generate_synthetic_cohort(
    ts_arr,
    aligned_grid,
    fpca_,
    fpca_grid,
    landmarks,
    rng,
    noise_multiplier,
    leak_mask=None,
    points_per_beat=100,
    original_length=1000,
):
    """Generate synthetic ECGs via FPCA perturbation, then reverse-align to the original axis.

    Landmarks are reused from the real traces under the assumption that FPC
    perturbation and residual swapping leave P/R/T locations essentially unchanged.
    """
    scores, residuals = transform_and_reconstruct(ts_arr, aligned_grid, fpca_, fpca_grid)
    
    # 1. Base perturbation for the entire cohort
    pert_sc = perturb_fpca_scores(scores, rng, noise_multiplier)
    
    # 2. LATENT OVERFITTING (The Grey Box Leak):
    # If a mask is provided, overwrite the perturbed scores with the EXACT real FPC scores.
    if leak_mask is not None:
        pert_sc[leak_mask] = scores[leak_mask]
    
    # # 3. Always apply donor-swapped residuals to mask the raw 1000d trace
    # donor_idx = rng.choice(len(residuals), size=len(scores), replace=True)
    
    # # CRITICAL: Prevent self-sampling for leaked records so the 1000d trace never matches perfectly
    # if leak_mask is not None:
    #     leaked_indices = np.where(leak_mask)[0]
    #     for i in leaked_indices:
    #         while donor_idx[i] == i:
    #             donor_idx[i] = rng.choice(len(residuals))
                
    # sampled_residuals = residuals[donor_idx]
    
    components = np.asarray(fpca_.components_.data_matrix, dtype=np.float64).reshape(fpca_.n_components, -1)
    mean = np.asarray(fpca_.mean_.data_matrix, dtype=np.float64).reshape(-1)
    
    flawed = (pert_sc @ components + mean) + residuals
    pert_aligned = resample_curves(flawed, fpca_grid, aligned_grid)
    # 4. Project back onto the original (e.g. 1000-sample) time axis.
    return reverse_align_ecg(
        pert_aligned,
        landmarks,
        points_per_beat=points_per_beat,
        original_length=original_length,
    )

def _valid_beats(landmark_row):
    landmark_row = np.asarray(landmark_row)
    return landmark_row[landmark_row[:, 1] >= 0]


def _resample_segment(seg, target_len):
    """Linearly resample a 1D segment onto ``target_len`` samples."""
    seg = np.asarray(seg, dtype=np.float64).reshape(-1)
    target_len = int(target_len)
    if target_len <= 0:
        return np.empty(0, dtype=np.float64)
    if seg.size == 0:
        return np.zeros(target_len, dtype=np.float64)
    if seg.size == target_len:
        return seg
    if seg.size == 1:
        return np.full(target_len, seg[0], dtype=np.float64)
    src = np.linspace(0.0, 1.0, seg.size)
    dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst, src, seg)


def _crossfade_segment(real_seg, synth_seg):
    """
    Linear cross-fade: start fully real, end fully synthetic.

    Real segment is resampled onto the synthetic segment length so the
    following synthetic suffix keeps its original timing.
    """
    synth_seg = np.asarray(synth_seg, dtype=np.float64).reshape(-1)
    if synth_seg.size == 0:
        return synth_seg
    real_on_synth = _resample_segment(real_seg, synth_seg.size)
    alpha = np.linspace(1.0, 0.0, synth_seg.size)
    return alpha * real_on_synth + (1.0 - alpha) * synth_seg


def _butterworth_baseline(x, sr, cutoff=0.5, order=2):
    """Low-pass Butterworth baseline; HF = x - baseline."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 12:
        return np.zeros_like(x)
    nyq = 0.5 * float(sr)
    wn = min(max(cutoff / nyq, 1e-6), 0.99)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)


def segment_leaking_creation(
    real_ts,
    real_landmark,
    synthetic_ts,
    synthetic_landmark,
    segment_leak_scale,
    real_grid,
    sr=None,
    baseline_cutoff=0.5,
):
    """
    Segment leak with dual-band cross-fading.

    Split each trace into baseline (Butterworth low-pass ~0.5 Hz) and high-frequency
    residual. For ``segment_leak_scale = n``:

      - HF: beats ``1..n`` from real; 1-beat cross-fade on beat ``n+1``; rest synthetic.
      - Baseline: 3-beat cross-fade over beats ``(n-1)..(n+1)`` (clamped at beat 1);
        real baseline before that window, synthetic baseline after.
      - Recombine baseline + HF, then project onto ``real_grid``.
    """
    rng = np.random.default_rng()
    if sr is None:
        sr = get_sr()

    real_ts = _as_curves(real_ts)
    synthetic_ts = _as_curves(synthetic_ts)
    real_landmark = np.asarray(real_landmark)
    synthetic_landmark = np.asarray(synthetic_landmark)
    real_grid = np.asarray(real_grid, dtype=np.float64).reshape(-1)

    n = int(segment_leak_scale)
    n_real = len(real_ts)
    n_synth = len(synthetic_ts)
    length = synthetic_ts.shape[1]
    out_len = len(real_grid)
    out = np.empty((n_synth, out_len), dtype=np.float64)

    def _project_to_real_grid(curve):
        curve = np.asarray(curve, dtype=np.float64).reshape(-1)
        if curve.size == out_len:
            return curve
        if curve.size < 2:
            curve = np.repeat(curve, 2)
        src_grid = np.linspace(real_grid[0], real_grid[-1], curve.size)
        fd = FDataGrid(data_matrix=curve[None, :], grid_points=src_grid)
        return np.asarray(fd(real_grid), dtype=np.float64).reshape(-1)

    if n <= 0:
        for i in range(n_synth):
            out[i] = _project_to_real_grid(synthetic_ts[i])
        return out

    # Need beat n (leaked) and beat n+1 (HF fade / end of baseline window).
    min_beats = n + 1
    # Baseline window: beats (n-1)..(n+1) -> 0-based indices [n-2, n], clamped.
    base_start = max(0, n - 2)
    base_end = n  # inclusive, beat n+1

    for i in range(n_synth):
        synth_beats = _valid_beats(synthetic_landmark[i])
        if len(synth_beats) < min_beats:
            out[i] = _project_to_real_grid(synthetic_ts[i])
            continue

        donor = None
        real_beats_j = None
        for _ in range(n_real):
            j = int(rng.integers(0, n_real))
            real_beats = _valid_beats(real_landmark[j])
            if len(real_beats) >= min_beats:
                donor = j
                real_beats_j = real_beats
                break
        if donor is None:
            out[i] = _project_to_real_grid(synthetic_ts[i])
            continue

        real_row = real_ts[donor]
        synth_row = synthetic_ts[i]

        real_xfade_p = int(real_beats_j[n, 0])
        real_xfade_t = int(real_beats_j[n, 2]) + 1
        synth_xfade_p = int(synth_beats[n, 0])
        synth_xfade_t = int(synth_beats[n, 2]) + 1

        real_win_p = int(real_beats_j[base_start, 0])
        real_win_t = int(real_beats_j[base_end, 2]) + 1
        synth_win_p = int(synth_beats[base_start, 0])
        synth_win_t = int(synth_beats[base_end, 2]) + 1

        if not (0 <= real_win_p <= real_xfade_p < real_xfade_t <= real_win_t <= real_row.shape[0]):
            out[i] = _project_to_real_grid(synth_row)
            continue
        if not (0 <= synth_win_p <= synth_xfade_p < synth_xfade_t <= synth_win_t <= length):
            out[i] = _project_to_real_grid(synth_row)
            continue

        real_base = _butterworth_baseline(real_row, sr, cutoff=baseline_cutoff)
        synth_base = _butterworth_baseline(synth_row, sr, cutoff=baseline_cutoff)
        real_hf = real_row - real_base
        synth_hf = synth_row - synth_base

        # --- Baseline: 3-beat (clamped) cross-fade on synth window timing ---
        base_prefix = real_base[:real_win_p]
        base_xfade = _crossfade_segment(
            real_base[real_win_p:real_win_t],
            synth_base[synth_win_p:synth_win_t],
        )
        base_suffix = synth_base[synth_win_t:]
        combined_base = np.concatenate([base_prefix, base_xfade, base_suffix])

        # --- HF: real through beat n; 1-beat cross-fade on beat n+1; synth after ---
        # Align HF window length to the baseline window (synth timing).
        hf_pre_win = _resample_segment(
            real_hf[real_win_p:real_xfade_p],
            synth_xfade_p - synth_win_p,
        )
        hf_fade = _crossfade_segment(
            real_hf[real_xfade_p:real_xfade_t],
            synth_hf[synth_xfade_p:synth_xfade_t],
        )
        hf_prefix = real_hf[:real_win_p]
        hf_suffix = synth_hf[synth_win_t:]
        combined_hf = np.concatenate([hf_prefix, hf_pre_win, hf_fade, hf_suffix])

        if combined_base.size != combined_hf.size:
            # Keep bands aligned; fall back if beat geometry is inconsistent.
            out[i] = _project_to_real_grid(synth_row)
            continue

        out[i] = _project_to_real_grid(combined_base + combined_hf)

    return out


if __name__ == "__main__":
    base_dir = Path("data/validation/privacy/")
    mia_dir = base_dir / "mia"
    reid_dir = base_dir / "reidentification"
    mia_dir.mkdir(parents=True, exist_ok=True)
    reid_dir.mkdir(parents=True, exist_ok=True)

    diagnostic = "NORM"
    lead = 1
    seed = 0
    n_flawed = get_flawed_size()
    n_components = get_fpca_n_components()
    points_per_beat = 100
    rng = np.random.default_rng(seed)

    print("Loading PTB-XL dataset...")
    raw_all_ts = load_real(diagnostic, lead)
    original_length = int(np.asarray(raw_all_ts).shape[-1])
    aligned_all_ts, kept_rows, aligned_grid, landmarks_all = align_dataset(
        raw_all_ts, get_sr(), points_per_beat=points_per_beat
    )
    # Real traces kept on the original axis so they match reverse-aligned synthetics.
    original_all_ts = _as_curves(raw_all_ts)[kept_rows]

    # 1. Split into Real (Audit Targets), Holdout Train (Shadow Train), Holdout Test (Shadow Val)
    real_idx, holdout_train_idx, holdout_test_idx = split_real_and_holdout(
        len(aligned_all_ts), rng, n_real=n_flawed, holdout_val_fraction=0.3
    )

    real_aligned = aligned_all_ts[real_idx]
    htrain_aligned = aligned_all_ts[holdout_train_idx]
    htest_aligned = aligned_all_ts[holdout_test_idx]

    real_ts = original_all_ts[real_idx]
    htrain_ts = original_all_ts[holdout_train_idx]
    htest_ts = original_all_ts[holdout_test_idx]

    real_landmarks = landmarks_all[real_idx]
    htrain_landmarks = landmarks_all[holdout_train_idx]
    htest_landmarks = landmarks_all[holdout_test_idx]

    print(f"Cohort sizes: Real (Audit) = {len(real_ts)}, Holdout Train = {len(htrain_ts)}, Holdout Test = {len(htest_ts)}")
    # Cohort sizes: Real (Audit) = 3000, Holdout Train = 2663, Holdout Test = 1141

    # 2. Fit FPCA exclusively on Holdout Train (aligned domain)
    print("Fitting FPCA on Holdout Train...")
    fpca_, fpca_grid = fit_fpca_model(htrain_aligned, aligned_grid, n_components)

    # 3. Build Shadow Datasets (Safe Anonymity Noise = 2.0 multiplier)
    print("Generating Shadow Datasets...")
    htrain_pert_ts = generate_synthetic_cohort(
        htrain_aligned, aligned_grid, fpca_, fpca_grid, htrain_landmarks, rng,
        noise_multiplier=2.0, points_per_beat=points_per_beat, original_length=original_length,
    )
    htest_pert_ts = generate_synthetic_cohort(
        htest_aligned, aligned_grid, fpca_, fpca_grid, htest_landmarks, rng,
        noise_multiplier=2.0, points_per_beat=points_per_beat, original_length=original_length,
    )
    
    # Near-exact Shadow datasets for Re-ID Positives (Breaching noise = 0.05)
    print("Generating near-exact Shadow datasets for Re-ID Positives...")
    htrain_reid_ts = generate_synthetic_cohort(
        htrain_aligned, aligned_grid, fpca_, fpca_grid, htrain_landmarks, rng,
        noise_multiplier=0.05, points_per_beat=points_per_beat, original_length=original_length,
    )
    htest_reid_ts = generate_synthetic_cohort(
        htest_aligned, aligned_grid, fpca_, fpca_grid, htest_landmarks, rng,
        noise_multiplier=0.05, points_per_beat=points_per_beat, original_length=original_length,
    )

    # Save Re-ID specifically formatted shadow data
    np.save(reid_dir / "shadow_train_real_ts.npy", htrain_ts)
    np.save(reid_dir / "shadow_train_synth_ts.npy", htrain_pert_ts)
    np.save(reid_dir / "shadow_train_reid_pos_ts.npy", htrain_reid_ts)
    
    np.save(reid_dir / "shadow_test_real_ts.npy", htest_ts)
    np.save(reid_dir / "shadow_test_synth_ts.npy", htest_pert_ts)
    np.save(reid_dir / "shadow_test_reid_pos_ts.npy", htest_reid_ts)

    # 4. Prepare MIA Datasets with Membership Labels
    # To detect latent overfitting, the MIA classifier must learn to distinguish 
    # between an overfitted (memorized) synthetic record and a safely generated one.
    # Shadow Train/Test: Leaked/Memorized records (noise=0.05) = Member (Label 1)
    #                    Safe perturbed records (noise=2.0) = Non-Member (Label 0)
    print("Preparing MIA Datasets...")
    mia_htrain_ts = np.concatenate([htrain_reid_ts, htrain_pert_ts], axis=0)
    mia_htrain_labels = np.concatenate([np.ones(len(htrain_reid_ts)), np.zeros(len(htrain_pert_ts))], axis=0)
    
    mia_htest_ts = np.concatenate([htest_reid_ts, htest_pert_ts], axis=0)
    mia_htest_labels = np.concatenate([np.ones(len(htest_reid_ts)), np.zeros(len(htest_pert_ts))], axis=0)

    np.save(mia_dir / "shadow_train_ts.npy", mia_htrain_ts)
    np.save(mia_dir / "shadow_train_labels.npy", mia_htrain_labels)
    np.save(mia_dir / "shadow_test_ts.npy", mia_htest_ts)
    np.save(mia_dir / "shadow_test_labels.npy", mia_htest_labels)

    # 5. Build Leaky Synthetic Target/Audit Datasets from Real cohort
    print("Generating Leaky Synthetic Target Datasets...")
    np.save(reid_dir / "audit_real_ts.npy", real_ts)

    subset_order = rng.permutation(len(real_aligned))
    leak_ratios = get_leak_ratios()

    for ratio in leak_ratios:
        print(f"Generating Leaky Synthetic Target Datasets for ratio {ratio}...")
        subset = subset_mask(len(real_aligned), ratio, subset_order)
        
        # Generate a cohort where the macro-features (FPC) are perfectly leaked, 
        # but the micro-features (1000d residuals) remain swapped.
        cur_ts = generate_synthetic_cohort(
            real_aligned, aligned_grid, fpca_, fpca_grid, real_landmarks, rng,
            noise_multiplier=2.0, leak_mask=subset,
            points_per_beat=points_per_beat, original_length=original_length,
        )
        
        name = f"synthetic_leaked_{ratio:.2f}"

        # Save for Re-ID (Candidate Queries)
        np.save(reid_dir / f"{name}_ts.npy", cur_ts)
        
        # Save for MIA: perfectly leaked samples are members (1), non-leaked are non-members (0).
        leaky_labels = np.zeros(len(cur_ts))
        leaky_labels[subset] = 1
        
        np.save(mia_dir / f"{name}_ts.npy", cur_ts)
        np.save(mia_dir / f"{name}_labels.npy", leaky_labels)

    # 6. Build Segment-Leaking Synthetic Target Datasets from Real cohort
    # Base release: safe FPCA synth (noise=2.0). Segment leak swaps the first
    # ``scale`` beats with a random real donor, then projects onto the original grid.
    print("Generating Segment-Leaking Synthetic Target Datasets...")
    segment_base_synth = generate_synthetic_cohort(
        real_aligned, aligned_grid, fpca_, fpca_grid, real_landmarks, rng,
        noise_multiplier=2.0,
        points_per_beat=points_per_beat, original_length=original_length,
    )
    # Original-axis grid for skfda projection inside segment_leaking_creation.
    real_grid = np.linspace(0.0, 1.0, original_length)

    for scale in get_segment_leak_scale():
        print(f"Generating Segment-Leaking datasets for scale={scale}...")
        # reverse_align reuses real landmarks, so synth landmarks == real_landmarks.
        segment_leaked_ts = segment_leaking_creation(
            real_ts,
            real_landmarks,
            segment_base_synth,
            real_landmarks,
            segment_leak_scale=scale,
            real_grid=real_grid,
        )
        name = f"synthetic_segment_leaked_{scale}"

        # Re-ID: segment-leaked queries against the real gallery.
        np.save(reid_dir / f"{name}_ts.npy", segment_leaked_ts)

        # MIA: segment-leaked = member (1), safe base synth = non-member (0).
        mia_segment_ts = np.concatenate([segment_leaked_ts, segment_base_synth], axis=0)
        mia_segment_labels = np.concatenate(
            [np.ones(len(segment_leaked_ts)), np.zeros(len(segment_base_synth))],
            axis=0,
        )
        np.save(mia_dir / f"{name}_ts.npy", mia_segment_ts)
        np.save(mia_dir / f"{name}_labels.npy", mia_segment_labels)
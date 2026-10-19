"""
Privacy-leaking dataset creation for validating the privacy attacks (AIA, re-identification).

Pipeline (all randomness driven by `seed`):
1. Load lead II of data/NORM_100.npy (N x 1000) and its row-aligned demographics
   data/NORM_100_demographics.pkl (N x 7, PTB-XL column order).
2. Holdout-real split: random 50-50 split by patient_id, so a patient with several
   records is entirely a member (real) or a non-member (holdout).
3. Fit FPCA on the landmark-aligned member ECGs, so score row i belongs to member i.
   A non-memorized series is that member's ECG reconstructed from a slight perturbation
   of those FPC scores, placed back on the original 1000 samples. A memorized series is
   the raw member ECG. Demographics of a non-memorized row are a slight perturbation of
   that same member. Identifiers stay exact, so the demographic-series link is unchanged.
4. For each leak ratio r, sample one subset of the real members (nested across ratios:
   the 0.10 subset is inside the 0.15 subset, and so on). Build four datasets from that
   subset, one per leak situation. Leaking means exact memorization of that modality.
5. Train-test split by strat_fold (PTB-XL folds are patient-disjoint): folds 9-10 are test.
   This yields 4 situations x 5 ratios = 20 synthetic datasets.

Outputs in data/validation/privacy/, for split in {train, test}:
    real_/holdout_{split}_demographics.npy
    real_/holdout_{split}_ts.npy
    real_/holdout_{split}_index.npy          row in NORM_100.npy
    synthetic_{split}_index.npy              source member of each synthetic row
    synthetic_subset_{r}_{split}.npy         bool, the shared sampled subset at ratio r
    synthetic_{situation}_{r}_{split}_demographics.npy
    synthetic_{situation}_{r}_{split}_ts.npy
    config.json

Usage:
    python -m scenario_engineering.privacy_leaking_dataset_creation
"""
import json
import os
from pathlib import Path

os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from skfda.representation.grid import FDataGrid

from preprocess.ptbxl_preprocess import align_ecg, get_demographic_columns, get_landmarks, get_sr, load_dataset, load_demographics
from transformation.fda.fpca import basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda, fpca_with_param

def get_leak_ratios():
    return [0.10, 0.15, 0.20, 0.25, 0.30]

def get_demographic_perturbation():
    # Non-memorized demographics: slight noise, sized like the series perturbation.
    # Sex is binary, so it is flipped with a small probability instead of Gaussian noise.
    return {"continuous_noise": 0.05, "continuous_columns": ["age", "height", "weight"], "sex_flip_prob": 0.05}

def get_series_noise():
    # Sigma of the FPC-score noise as a multiple of that component's score std.
    # 0.05 matches the slight raw-space perturbation this replaces.
    return 0.05

def get_fpca_n_components():
    return 10

def get_leak_situations():
    """Four datasets built from the same sampled subset. Leaking is exact memorization."""
    return [
        {"name": "demo_not_leaked_series_not_leaked", "demo_leaked": False, "series_leaked": False},
        {"name": "demo_not_leaked_series_leaked", "demo_leaked": False, "series_leaked": True},
        {"name": "demo_leaked_series_not_leaked", "demo_leaked": True, "series_leaked": False},
        {"name": "demo_leaked_series_leaked", "demo_leaked": True, "series_leaked": True},
    ]

# PTB-XL encodes ages above 89 as 300.
max_age = 90

def load_real(diagnostic, lead):
    """Real time series and demographics, row for row."""
    ts = load_dataset(diagnostic=diagnostic, sampling_rate=get_sr(), lead=lead)
    demographics = pd.DataFrame(load_demographics(diagnostic, get_sr()), columns=get_demographic_columns())
    demographics = demographics.apply(pd.to_numeric, errors="coerce")
    if len(ts) != len(demographics):
        raise ValueError(f"{len(ts)} time series but {len(demographics)} demographic rows.")
    demographics["age"] = demographics["age"].clip(upper=max_age)
    return ts, demographics

def holdout_real_split(demographics, rng, real_fraction=0.5):
    """Patient-level random split into real (member) and holdout (non-member) row indices."""
    patients = rng.permutation(demographics["patient_id"].unique())
    real_patients = patients[:int(round(real_fraction * len(patients)))]
    is_real = demographics["patient_id"].isin(real_patients).to_numpy()
    return np.where(is_real)[0], np.where(~is_real)[0]

def train_test_split_by_fold(demographics, test_folds):
    is_test = demographics["strat_fold"].isin(test_folds).to_numpy()
    return np.where(~is_test)[0], np.where(is_test)[0]

def leak_demographics(demographics, rng, continuous_noise, continuous_columns, sex_flip_prob):
    """Perturbed copy of the member demographics; identifiers are kept so the leak stays linkable."""
    leaked = demographics.copy().reset_index(drop=True)
    for col in continuous_columns:
        std = leaked[col].std()
        leaked[col] = leaked[col] + rng.normal(0, continuous_noise * std, size=len(leaked))
    leaked["age"] = leaked["age"].clip(0, max_age)
    leaked[["height", "weight"]] = leaked[["height", "weight"]].clip(lower=1)
    flip = rng.random(len(leaked)) < sex_flip_prob
    leaked.loc[flip, "sex"] = 1 - leaked.loc[flip, "sex"]
    return leaked

def subset_mask(n, ratio, order):
    """First `ratio` of `order`. Larger ratios extend the same order, so the subsets are nested."""
    mask = np.zeros(n, dtype=bool)
    mask[order[:int(round(ratio * n))]] = True
    return mask

def scenario_demographics(member_demographics, perturbed_demographics, subset, demo_leaked):
    demographics = perturbed_demographics.copy().reset_index(drop=True)
    if demo_leaked:
        member = member_demographics.reset_index(drop=True)
        rows = np.flatnonzero(subset)
        demographics.iloc[rows] = member.iloc[rows].to_numpy()
    return demographics

def scenario_time_series(member_ts, perturbed_ts, subset, series_leaked):
    ts = perturbed_ts.astype(member_ts.dtype, copy=True)
    if series_leaked:
        ts[subset] = member_ts[subset]
    return ts

def landmarks_per_signal(signals, sr):
    """Landmarks for every signal that has a valid beat. Dropped rows stay out so indices stay aligned."""
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

def fit_member_fpca(member_ts, n_components, sr, domain_range=(0, 1)):
    """Landmark-aligned FPCA of the member ECGs. Score row j belongs to kept member kept[j]."""
    landmarks, kept = landmarks_per_signal(member_ts, sr)
    fd = align_ecg(member_ts[kept], landmarks)
    if fd.n_samples != len(kept):
        raise ValueError("align_ecg dropped signals; rows would no longer match their demographics.")
    n_basis = int(fd.data_matrix.shape[1] / 2)
    lambda_ = basis_smoothing_hyperparameter_tuning(fd, n_basis, domain_range)
    fd_smooth, _, _, _ = basis_smoothing_with_lambda(fd, lambda_, n_basis, domain_range)
    _, _, _, var_ratio, fpca_ = fpca_with_param(fd_smooth, n_components)
    grid = np.asarray(fpca_.components_.grid_points[0], dtype=np.float64)
    fd_smooth = FDataGrid(data_matrix=fd_smooth(grid), grid_points=grid)
    scores = np.asarray(fpca_.transform(fd_smooth), dtype=np.float64)
    return fpca_, scores, np.asarray(var_ratio, dtype=np.float64), landmarks, kept, grid

def reconstruct_ecg(fpca_, scores):
    """Curve = mean + FPC scores @ FPCs, one row per score row, on the FPCA grid."""
    components = np.asarray(fpca_.components_.data_matrix, dtype=np.float64).reshape(fpca_.n_components, -1)
    mean = np.asarray(fpca_.mean_.data_matrix, dtype=np.float64).reshape(-1)
    return scores @ components + mean

def perturb_fpca_scores(scores, rng, noise_multiplier):
    """Slight Gaussian noise on each member's FPC scores. Row order is unchanged."""
    std = np.std(scores, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return scores + rng.normal(0.0, noise_multiplier * std, size=scores.shape)

def place_reconstruction(member_ts, landmarks, kept, curves, grid):
    """Put each reconstructed aligned curve back on that member's original 1000 samples."""
    beat_counts = [int(np.sum(landmark[:, 1] >= 0)) for landmark in landmarks]
    max_beats = max(beat_counts)
    phase_scale = max_beats * 2 * np.pi
    placed = np.array(member_ts, dtype=np.float64, copy=True)
    for row, member_index in enumerate(kept):
        landmark = landmarks[row]
        landmark = landmark[landmark[:, 1] >= 0]
        start = int(landmark[0, 0])
        end = int(landmark[-1, 2])
        known_indices, known_phases = [], []
        for beat_index, (p_onset, r_peak, t_offset) in enumerate(landmark):
            known_indices.extend([int(p_onset) - start, int(r_peak) - start, int(t_offset) - start])
            known_phases.extend([
                beat_index * 2 * np.pi,
                beat_index * 2 * np.pi + np.pi,
                beat_index * 2 * np.pi + 2 * np.pi,
            ])
        phase = np.interp(np.arange(end - start + 1), known_indices, known_phases)
        placed[member_index, start:end + 1] = np.interp(phase / phase_scale, grid, curves[row])
    return placed

def save_split(save_path, name, split_idx, demographics, index, ts=None):
    for split, idx in zip(("train", "test"), split_idx):
        np.save(save_path / f"{name}_{split}_demographics.npy", demographics.iloc[idx].to_numpy(dtype=np.float64))
        np.save(save_path / f"{name}_{split}_index.npy", index[idx])
        if ts is not None:
            np.save(save_path / f"{name}_{split}_ts.npy", ts[idx])

if __name__ == "__main__":
    save_path = Path("data/validation/privacy/")
    save_path.mkdir(parents=True, exist_ok=True)

    diagnostic = "NORM"
    lead = 1
    seed = 0
    test_folds = [9, 10]
    rng = np.random.default_rng(seed)

    real_all_ts, real_all_demographics = load_real(diagnostic, lead)

    # Holdout-Real Split
    real_idx, holdout_idx = holdout_real_split(real_all_demographics, rng)
    real_ts, real_demographics = real_all_ts[real_idx], real_all_demographics.iloc[real_idx].reset_index(drop=True)
    holdout_ts = real_all_ts[holdout_idx]
    holdout_demographics = real_all_demographics.iloc[holdout_idx].reset_index(drop=True)

    # One perturbation of every member. Leaked modalities replace this with an exact copy.
    # Series noise is applied to that member's FPC scores, then the ECG is reconstructed.
    perturbed_demographics = leak_demographics(real_demographics, rng, **get_demographic_perturbation())
    n_components = get_fpca_n_components()
    fpca_, scores, var_ratio, landmarks, kept, grid = fit_member_fpca(real_ts, n_components, get_sr())
    perturbed_scores = perturb_fpca_scores(scores, rng, get_series_noise())
    curves = reconstruct_ecg(fpca_, perturbed_scores)
    perturbed_ts = place_reconstruction(real_ts, landmarks, kept, curves, grid)
    if perturbed_ts.shape != real_ts.shape:
        raise ValueError(f"Reconstructed series shape {perturbed_ts.shape} does not match members {real_ts.shape}.")
    subset_order = rng.permutation(len(real_ts))

    # Train-Test Split by strat_fold; synthetic rows inherit their source member's fold
    real_split = train_test_split_by_fold(real_demographics, test_folds)
    holdout_split = train_test_split_by_fold(holdout_demographics, test_folds)
    synthetic_split = real_split

    for stale in save_path.glob("synthetic_*"):
        stale.unlink()
    save_split(save_path, "real", real_split, real_demographics, real_idx, real_ts)
    save_split(save_path, "holdout", holdout_split, holdout_demographics, holdout_idx, holdout_ts)
    for split, idx in zip(("train", "test"), synthetic_split):
        np.save(save_path / f"synthetic_{split}_index.npy", real_idx[idx])

    # Four leak situations from the same nested subset, at each ratio
    situations = get_leak_situations()
    for ratio in get_leak_ratios():
        subset = subset_mask(len(real_ts), ratio, subset_order)
        for split, idx in zip(("train", "test"), synthetic_split):
            np.save(save_path / f"synthetic_subset_{ratio:.2f}_{split}.npy", subset[idx])
        for situation in situations:
            demographics = scenario_demographics(real_demographics, perturbed_demographics, subset, situation["demo_leaked"])
            ts = scenario_time_series(real_ts, perturbed_ts, subset, situation["series_leaked"])
            name = f"synthetic_{situation['name']}_{ratio:.2f}"
            for split, idx in zip(("train", "test"), synthetic_split):
                np.save(save_path / f"{name}_{split}_demographics.npy", demographics.iloc[idx].to_numpy(dtype=np.float64))
                np.save(save_path / f"{name}_{split}_ts.npy", ts[idx])

    config = {
        "diagnostic": diagnostic,
        "lead": lead,
        "seed": seed,
        "test_folds": test_folds,
        "max_age": max_age,
        "demographic_columns": get_demographic_columns(),
        "demographic_perturbation": get_demographic_perturbation(),
        "series_perturbation": "fpca_score_noise",
        "fpca_n_components": n_components,
        "fpca_n_fit": int(len(kept)),
        "fpca_n_fit_dropped": int(len(real_ts) - len(kept)),
        "fpca_var_ratio": var_ratio.tolist(),
        "series_noise_multiplier": get_series_noise(),
        "leak_ratios": get_leak_ratios(),
        "leak_situations": situations,
        "sizes": {
            name: {"train": len(split[0]), "test": len(split[1])}
            for name, split in [("real", real_split), ("holdout", holdout_split)]
        },
    }
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config["sizes"], indent=2))

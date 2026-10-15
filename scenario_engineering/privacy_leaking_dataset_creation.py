"""
Privacy-leaking dataset creation for validating the privacy attacks (AIA, re-identification).

Pipeline (all randomness driven by `seed`):
1. Load lead II of data/NORM_100.npy (N x 1000) and its row-aligned demographics
   data/NORM_100_demographics.pkl (N x 7, PTB-XL column order).
2. Holdout-real split: random 50-50 split by patient_id, so a patient with several
   records is entirely a member (real) or a non-member (holdout).
3. Synthetic demographics (single leaking scale): one row per real member, copying its
   demographics (patient_id, ecg_id and strat_fold kept) and perturbing age/height/weight
   with Gaussian noise and flipping sex.
4. Synthetic time series (multiple leaking scales, full-series memorization): a fraction
   `ratio` of synthetic rows carry their source member's exact series; the remaining rows
   carry series sampled from the generator output (NORM rows of data/synthetic_final.npy).
   Leaked rows are nested across ratios (rows leaked at 0.10 are also leaked at 0.15, ...).
5. Train-test split by strat_fold (PTB-XL folds are patient-disjoint): folds 9-10 are test.

Outputs in data/validation/privacy/, for set in {real, holdout, synthetic} and split in {train, test}:
    {set}_{split}_demographics.npy        float64, N x 7 in `get_demographic_columns()` order
    {set}_{split}_index.npy               row in NORM_100.npy (for synthetic: the source member)
    real_/holdout_{split}_ts.npy          N x 1000
    synthetic_{split}_ts_leak_{r}.npy     N x 1000
    synthetic_{split}_leak_mask_{r}.npy   bool, True where the series is memorized
    config.json                           parameters and set sizes

Usage:
    python -m scenario_engineering.privacy_leaking_dataset_creation
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess.ptbxl_preprocess import get_demographic_columns, get_sr, load_dataset, load_demographics, \
    load_synthetic_dataset

def get_leak_ratios():
    return [0.10, 0.15, 0.20, 0.25, 0.30]

def get_demographic_perturbation():
    # Noise sigma as a multiple of the member std for continuous columns; flip probability for sex.
    return {"continuous_noise": 0.25, "continuous_columns": ["age", "height", "weight"], "sex_flip_prob": 0.10}

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

def leak_time_series(member_ts, generated_ts, leak_order, ratio):
    """Full-series memorization: the first `ratio` of rows in `leak_order` copy the member series."""
    n_leak = int(round(ratio * len(member_ts)))
    leak_mask = np.zeros(len(member_ts), dtype=bool)
    leak_mask[leak_order[:n_leak]] = True
    ts = generated_ts.astype(member_ts.dtype)
    ts[leak_mask] = member_ts[leak_mask]
    return ts, leak_mask

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

    # Privacy-Leaking Demographics (single scale), one synthetic row per real member
    synthetic_demographics = leak_demographics(real_demographics, rng, **get_demographic_perturbation())

    # Non-leaking series from the generator, sampled without replacement
    generated_all = load_synthetic_dataset(diagnostic, lead)
    if len(generated_all) < len(real_ts):
        raise ValueError(f"Only {len(generated_all)} generated series for {len(real_ts)} synthetic rows.")
    generated_idx = rng.choice(len(generated_all), size=len(real_ts), replace=False)
    generated_ts = generated_all[generated_idx]
    leak_order = rng.permutation(len(real_ts))

    # Train-Test Split by strat_fold; synthetic rows inherit their source member's fold
    real_split = train_test_split_by_fold(real_demographics, test_folds)
    holdout_split = train_test_split_by_fold(holdout_demographics, test_folds)
    synthetic_split = real_split

    save_split(save_path, "real", real_split, real_demographics, real_idx, real_ts)
    save_split(save_path, "holdout", holdout_split, holdout_demographics, holdout_idx, holdout_ts)
    save_split(save_path, "synthetic", synthetic_split, synthetic_demographics, real_idx)

    # Privacy-Leaking Time Series (full-series memorization at each ratio)
    for ratio in get_leak_ratios():
        synthetic_ts, leak_mask = leak_time_series(real_ts, generated_ts, leak_order, ratio)
        for split, idx in zip(("train", "test"), synthetic_split):
            np.save(save_path / f"synthetic_{split}_ts_leak_{ratio:.2f}.npy", synthetic_ts[idx])
            np.save(save_path / f"synthetic_{split}_leak_mask_{ratio:.2f}.npy", leak_mask[idx])

    config = {
        "diagnostic": diagnostic,
        "lead": lead,
        "seed": seed,
        "test_folds": test_folds,
        "max_age": max_age,
        "demographic_columns": get_demographic_columns(),
        "demographic_perturbation": get_demographic_perturbation(),
        "leak_ratios": get_leak_ratios(),
        "generated_source_index": generated_idx.tolist(),
        "sizes": {
            name: {"train": len(split[0]), "test": len(split[1])}
            for name, split in [("real", real_split), ("holdout", holdout_split), ("synthetic", synthetic_split)]
        },
    }
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config["sizes"], indent=2))

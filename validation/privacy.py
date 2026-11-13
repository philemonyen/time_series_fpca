"""
Orchestration of Time-Series Privacy Attacks (MIA & Re-identification).

Stored ECG cohorts are unaligned (wall-clock time). Blackbox attacks operate on
those raw traces. Greybox I/II need FPCA / DMap features, so holdout curves are
landmark-aligned before FPCA is fit, and every real/synthetic cohort is aligned
onto the same phase grid before FPCA / DMap transform.

1. Align holdout ECGs; fit FPCA and DMap on the aligned holdout.
2. Keep unaligned TS for blackbox; derive feat/dmap from aligned TS for greybox.
3. Train ReID / end-to-end MIA models (Blackbox, Greybox I, Greybox II).
   MIA is a single BCE path: modality embeddings + MLP classifier
   (kwargs: epochs / batch_size / patience, seed=).
4. MIA: train on shadow data, evaluate on leaky synth only (AUC).
   Re-ID: audit leaky synth queries against the real gallery.
5. Repeat for FPC leak-ratio and segment-leak-scale releases.
6. Export scores to images/privacy_val/privacy_val_result_ts_gb2.json.
"""
import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import numpy as np
from pathlib import Path
from skfda.representation.grid import FDataGrid
from transformation.fda.fpca import (
    fpca_with_param, 
    basis_smoothing_hyperparameter_tuning, 
    basis_smoothing_with_lambda, 
    fpca_hyperparameter_tuning
)

from transformation.nonlinear.diffusion_map import DenseDiffusionMap

from privacy_attack.mia import attacks as mia_attacks
from privacy_attack.reidentification import attacks as reid_attacks
from scenario_engineering.privacy_leaking_dataset_creation import (
    get_leak_ratios,
    get_segment_leak_scale,
)
from preprocess.ptbxl_preprocess import align_ecg, get_landmarks, get_sr

data_path_mia = Path("data/validation/privacy/mia/")
data_path_reid = Path("data/validation/privacy/reidentification/")
results_path = Path("images/privacy_val/")
results_path.mkdir(parents=True, exist_ok=True)

POINTS_PER_BEAT = 100


def _as_curves(ts):
    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim == 3:
        ts = ts.reshape(ts.shape[0], -1)
    return ts


def as_aligned_grid(ts, n_timepoints=None):
    ts = _as_curves(ts)
    if n_timepoints is not None and ts.shape[1] != n_timepoints:
        raise ValueError(f"Expected aligned length {n_timepoints}, got {ts.shape[1]}.")
    grid = np.linspace(0, 1, ts.shape[1])
    return FDataGrid(data_matrix=ts, grid_points=grid)


def align_batch(ts, sr, points_per_beat=POINTS_PER_BEAT, max_beats=None):
    """
    Landmark-align unaligned ECGs onto a shared phase grid.

    Returns one aligned curve per input row (index-aligned with labels / blackbox TS).
    Signals without valid landmarks fall back to linear resampling onto the phase grid.
    Pass ``max_beats`` from a prior fit so transform batches match the FPCA grid length.
    """
    ts = _as_curves(ts)
    n = len(ts)
    per_signal_lm = []
    beat_counts = []
    for signal in ts:
        try:
            lm = get_landmarks(signal[None], sr)
            if len(lm) == 0:
                per_signal_lm.append(None)
                continue
            beats = lm[0]
            beats = beats[beats[:, 1] >= 0]
            if len(beats) == 0:
                per_signal_lm.append(None)
                continue
            per_signal_lm.append(beats)
            beat_counts.append(len(beats))
        except Exception:
            per_signal_lm.append(None)

    if max_beats is None:
        if not beat_counts:
            raise ValueError("No ECG had valid landmarks for alignment.")
        max_beats = max(beat_counts)

    total_points = max_beats * points_per_beat
    padded = np.full((n, max_beats, 3), -1, dtype=int)
    valid = np.zeros(n, dtype=bool)
    for i, beats in enumerate(per_signal_lm):
        if beats is None:
            continue
        valid[i] = True
        n_b = min(len(beats), max_beats)
        padded[i, :n_b] = beats[:n_b]

    out = np.zeros((n, total_points), dtype=np.float64)
    source_grid = np.linspace(0.0, 1.0, ts.shape[1])
    target_grid = np.linspace(0.0, 1.0, total_points)

    def _linear_fallback(row):
        return np.interp(target_grid, source_grid, row)

    # Align one curve at a time so a single bad landmark cannot drop rows or
    # desynchronize out[valid] vs align_ecg's silently-skipped outputs.
    for i in range(n):
        if not valid[i]:
            out[i] = _linear_fallback(ts[i])
            continue
        try:
            fd = align_ecg(
                ts[i:i + 1],
                padded[i:i + 1],
                points_per_beat=points_per_beat,
                max_beats=max_beats,
            )
            out[i] = _as_curves(fd.data_matrix)[0]
        except Exception:
            valid[i] = False
            out[i] = _linear_fallback(ts[i])
    return out, max_beats


class FPCAFeatures:
    def __init__(self, domain_range=(0, 1)):
        self.domain_range = domain_range

    def fit(self, ts):
        fd = as_aligned_grid(ts)
        _, self.n_timepoints, _ = fd.data_matrix.shape
        self.n_basis = int(self.n_timepoints / 2)
        lambda_ = basis_smoothing_hyperparameter_tuning(fd, self.n_basis, self.domain_range)
        fd_smooth, _, _, _ = basis_smoothing_with_lambda(fd, lambda_, self.n_basis, self.domain_range)
        n_components = fpca_hyperparameter_tuning(fd_smooth)
        _, _, _, self.var_ratio, self.fpca_ = fpca_with_param(fd_smooth, n_components)
        self.grid = self.fpca_.components_.grid_points[0]
        return self

    def transform(self, ts):
        fd = as_aligned_grid(ts, self.n_timepoints)
        lambda_ = basis_smoothing_hyperparameter_tuning(fd, self.n_basis, self.domain_range)
        fd_smooth, _, _, _ = basis_smoothing_with_lambda(fd, lambda_, self.n_basis, self.domain_range)
        fd_smooth = FDataGrid(data_matrix=fd_smooth(self.grid), grid_points=self.grid)
        return np.asarray(self.fpca_.transform(fd_smooth), dtype=np.float64)


mia_metrics = ("auc", "precision", "recall", "f1")
reid_metrics = (
    "top1_accuracy", "top5_accuracy", "top10_accuracy",
    "top1_lift", "top5_lift", "top10_lift", "mrr",
    "top1_accuracy_leaky", "top5_accuracy_leaky", "top10_accuracy_leaky", "mrr_leaky",
)

def summarize(runs, metric_names):
    return {
        "mean": {k: float(np.mean([r[k] for r in runs])) for k in metric_names},
        "std": {k: float(np.std([r[k] for r in runs])) for k in metric_names},
        "runs": [{k: r[k] for k in metric_names} for r in runs],
    }

def run_mia_evaluation(attack_cls, train_ts, train_feat, train_dmap, train_labels,
                       val_ts, val_feat, val_dmap, val_labels,
                       audit_ts, audit_feat, audit_dmap, audit_labels,
                       seeds, metric_names, **attack_kwargs):
    """Fit/evaluate end-to-end MIA (embedding branches + MLP head, one gradient path)."""
    runs = []
    for seed in seeds:
        attack = attack_cls(seed=seed, **attack_kwargs)
        attack.fit(
            train_ts, train_feat, train_dmap, train_labels,
            val_ts, val_feat, val_dmap, val_labels,
        )
        runs.append(attack.evaluate(audit_ts, audit_feat, audit_dmap, audit_labels))
    return summarize(runs, metric_names)

def run_reid_evaluation(attack_cls, shadow_train, shadow_val, audit_real, audit_synth, seeds, metric_names, **attack_kwargs):
    runs = []
    for seed in seeds:
        attack = attack_cls(seed=seed, **attack_kwargs)
        attack.fit(
            shadow_train["real_ts"], shadow_train["real_feat"], shadow_train["real_dmap"],
            shadow_train["synth_ts"], shadow_train["synth_feat"], shadow_train["synth_dmap"],
            train_pos_ts=shadow_train["pos_ts"], train_pos_feat=shadow_train["pos_feat"], train_pos_dmap=shadow_train["pos_dmap"],
            val_real_ts=shadow_val["real_ts"], val_real_feat=shadow_val["real_feat"], val_real_dmap=shadow_val["real_dmap"],
            val_synth_ts=shadow_val["synth_ts"], val_synth_feat=shadow_val["synth_feat"], val_synth_dmap=shadow_val["synth_dmap"],
            val_pos_ts=shadow_val["pos_ts"], val_pos_feat=shadow_val["pos_feat"], val_pos_dmap=shadow_val["pos_dmap"]
        )
        runs.append(attack.evaluate(
            audit_real["ts"], audit_real["feat"], audit_real["dmap"],
            audit_synth["ts"], audit_synth["feat"], audit_synth["dmap"],
            leak_labels=audit_synth.get("labels"),
        ))
    return summarize(runs, metric_names)

if __name__ == "__main__":
    domain_range = (0, 1)
    seeds = [0, 1, 2]
    sr = get_sr()
    
    # End-to-end MIA (privacy_attack.mia): NN training hyperparameters, not RF.
    mia_kwargs = {"epochs": 100, "batch_size": 64, "patience": 15}
    reid_kwargs = {"epochs": 100, "batch_size": 64, "patience": 15, "margin": 1.0}

    print("Aligning holdout ECGs, then fitting FPCA and DMap...")
    holdout_train_ts = np.load(data_path_reid / "shadow_train_real_ts.npy")
    holdout_test_ts = np.load(data_path_reid / "shadow_test_real_ts.npy")

    holdout_train_aligned, max_beats = align_batch(holdout_train_ts, sr, POINTS_PER_BEAT)
    holdout_test_aligned, _ = align_batch(
        holdout_test_ts, sr, POINTS_PER_BEAT, max_beats=max_beats
    )
    full_holdout_aligned = np.concatenate([holdout_train_aligned, holdout_test_aligned], axis=0)

    fpca = FPCAFeatures(domain_range).fit(full_holdout_aligned)
    holdout_scores = fpca.transform(full_holdout_aligned)
    
    # Fit the custom DenseDiffusionMap on the FPC scores of the holdout
    dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(holdout_scores)

    def feat_from_unaligned(ts):
        aligned, _ = align_batch(ts, sr, POINTS_PER_BEAT, max_beats=max_beats)
        return fpca.transform(aligned)

    print("Loading Re-Identification Datasets...")
    # 1. Prepare Re-ID Shadow Training and Validation cohorts
    # real_ts / synth_ts / pos_ts stay unaligned for blackbox; feat/dmap use aligned FPCA.
    reid_shadow_train = {
        "real_ts": holdout_train_ts,
        "real_feat": fpca.transform(holdout_train_aligned),
        "synth_ts": np.load(data_path_reid / "shadow_train_synth_ts.npy"),
        "pos_ts": np.load(data_path_reid / "shadow_train_reid_pos_ts.npy"),
    }
    reid_shadow_train["real_dmap"] = dmap.transform(reid_shadow_train["real_feat"])
    reid_shadow_train["synth_feat"] = feat_from_unaligned(reid_shadow_train["synth_ts"])
    reid_shadow_train["synth_dmap"] = dmap.transform(reid_shadow_train["synth_feat"])
    reid_shadow_train["pos_feat"] = feat_from_unaligned(reid_shadow_train["pos_ts"])
    reid_shadow_train["pos_dmap"] = dmap.transform(reid_shadow_train["pos_feat"])

    reid_shadow_val = {
        "real_ts": holdout_test_ts,
        "real_feat": fpca.transform(holdout_test_aligned),
        "synth_ts": np.load(data_path_reid / "shadow_test_synth_ts.npy"),
        "pos_ts": np.load(data_path_reid / "shadow_test_reid_pos_ts.npy"),
    }
    reid_shadow_val["real_dmap"] = dmap.transform(reid_shadow_val["real_feat"])
    reid_shadow_val["synth_feat"] = feat_from_unaligned(reid_shadow_val["synth_ts"])
    reid_shadow_val["synth_dmap"] = dmap.transform(reid_shadow_val["synth_feat"])
    reid_shadow_val["pos_feat"] = feat_from_unaligned(reid_shadow_val["pos_ts"])
    reid_shadow_val["pos_dmap"] = dmap.transform(reid_shadow_val["pos_feat"])

    # Prepare Re-ID Audit Real Targets
    reid_audit_real = {
        "ts": np.load(data_path_reid / "audit_real_ts.npy"),
    }
    reid_audit_real["feat"] = feat_from_unaligned(reid_audit_real["ts"])
    reid_audit_real["dmap"] = dmap.transform(reid_audit_real["feat"])

    print("Loading Membership Inference Datasets...")
    # 2. Prepare MIA Datasets
    mia_train_ts = np.load(data_path_mia / "shadow_train_ts.npy")
    mia_train_labels = np.load(data_path_mia / "shadow_train_labels.npy")
    mia_train_feat = feat_from_unaligned(mia_train_ts)
    mia_train_dmap = dmap.transform(mia_train_feat)

    mia_val_ts = np.load(data_path_mia / "shadow_test_ts.npy")
    mia_val_labels = np.load(data_path_mia / "shadow_test_labels.npy")
    mia_val_feat = feat_from_unaligned(mia_val_ts)
    mia_val_dmap = dmap.transform(mia_val_feat)

    leak_ratios = get_leak_ratios()
    segment_scales = get_segment_leak_scale()

    result_tracking = {
        "config": {
            "seeds": seeds,
            "mia_kwargs": mia_kwargs,
            "reid_kwargs": reid_kwargs,
            "leak_ratios": leak_ratios,
            "segment_leak_scales": segment_scales,
            "points_per_beat": POINTS_PER_BEAT,
            "max_beats": int(max_beats),
            "n_audit_real": len(reid_audit_real["ts"]),
            "n_shadow_train": len(reid_shadow_train["real_ts"]),
            "n_shadow_val": len(reid_shadow_val["real_ts"]),
        },
        "leak_ratio": {},
        "segment_leak": {},
    }

    print("Beginning time-series privacy auditing across leak ratios...")

    for ratio in leak_ratios:
        ratio_key = f"{ratio:.2f}"
        stem = f"synthetic_leaked_{ratio_key}"

        # --- Re-ID Setup ---
        # Labels (1=leaky) live under mia/; same index mask as reid synth cohort.
        reid_audit_synth = {
            "ts": np.load(data_path_reid / f"{stem}_ts.npy"),
            "labels": np.load(data_path_mia / f"{stem}_labels.npy"),
        }
        reid_audit_synth["feat"] = feat_from_unaligned(reid_audit_synth["ts"])
        reid_audit_synth["dmap"] = dmap.transform(reid_audit_synth["feat"])

        # --- MIA Setup: evaluate on leaky synth only (no real-pool concat) ---
        mia_eval_ts = np.load(data_path_mia / f"{stem}_ts.npy")
        mia_eval_labels = np.load(data_path_mia / f"{stem}_labels.npy")
        mia_eval_feat = feat_from_unaligned(mia_eval_ts)
        mia_eval_dmap = dmap.transform(mia_eval_feat)

        result_tracking["leak_ratio"][ratio_key] = {"mia": {}, "reidentification": {}}

        print(f"[ratio {ratio_key}] Evaluating MIA...")
        for attack_name, attack_cls in mia_attacks.items():
            result_tracking["leak_ratio"][ratio_key]["mia"][attack_name] = run_mia_evaluation(
                attack_cls, mia_train_ts, mia_train_feat, mia_train_dmap, mia_train_labels,
                mia_val_ts, mia_val_feat, mia_val_dmap, mia_val_labels,
                mia_eval_ts, mia_eval_feat, mia_eval_dmap, mia_eval_labels,
                seeds, mia_metrics, **mia_kwargs
            )

        print(f"[ratio {ratio_key}] Evaluating Re-Identification...")
        for attack_name, attack_cls in reid_attacks.items():
            result_tracking["leak_ratio"][ratio_key]["reidentification"][attack_name] = run_reid_evaluation(
                attack_cls, reid_shadow_train, reid_shadow_val, reid_audit_real, reid_audit_synth,
                seeds, reid_metrics, **reid_kwargs
            )

        print(f"[Ratio {ratio_key}] Completed MIA and Re-ID evaluations.")

        with open(results_path / "privacy_val_result_ts_gb2.json", "w") as f:
            json.dump(result_tracking, f, indent=2)

    print("Beginning time-series privacy auditing across segment-leak scales...")

    for scale in segment_scales:
        scale_key = str(scale)
        stem = f"synthetic_segment_leaked_{scale}"

        # Re-ID queries are segment-leaked only; every query is risk-relevant.
        reid_audit_synth = {
            "ts": np.load(data_path_reid / f"{stem}_ts.npy"),
        }
        reid_audit_synth["labels"] = np.ones(len(reid_audit_synth["ts"]), dtype=np.float64)
        reid_audit_synth["feat"] = feat_from_unaligned(reid_audit_synth["ts"])
        reid_audit_synth["dmap"] = dmap.transform(reid_audit_synth["feat"])

        # MIA eval set already mixes segment-leaked (1) and safe base synth (0).
        mia_eval_ts = np.load(data_path_mia / f"{stem}_ts.npy")
        mia_eval_labels = np.load(data_path_mia / f"{stem}_labels.npy")
        mia_eval_feat = feat_from_unaligned(mia_eval_ts)
        mia_eval_dmap = dmap.transform(mia_eval_feat)

        result_tracking["segment_leak"][scale_key] = {"mia": {}, "reidentification": {}}

        print(f"[segment scale={scale_key}] Evaluating MIA...")
        for attack_name, attack_cls in mia_attacks.items():
            result_tracking["segment_leak"][scale_key]["mia"][attack_name] = run_mia_evaluation(
                attack_cls, mia_train_ts, mia_train_feat, mia_train_dmap, mia_train_labels,
                mia_val_ts, mia_val_feat, mia_val_dmap, mia_val_labels,
                mia_eval_ts, mia_eval_feat, mia_eval_dmap, mia_eval_labels,
                seeds, mia_metrics, **mia_kwargs
            )

        print(f"[segment scale={scale_key}] Evaluating Re-Identification...")
        for attack_name, attack_cls in reid_attacks.items():
            result_tracking["segment_leak"][scale_key]["reidentification"][attack_name] = run_reid_evaluation(
                attack_cls, reid_shadow_train, reid_shadow_val, reid_audit_real, reid_audit_synth,
                seeds, reid_metrics, **reid_kwargs
            )

        print(f"[Segment scale {scale_key}] Completed MIA and Re-ID evaluations.")

        with open(results_path / "privacy_val_result_ts_gb2.json", "w") as f:
            json.dump(result_tracking, f, indent=2)

    print(f"Audit results successfully written to {results_path / 'privacy_val_result_ts_gb2.json'}")

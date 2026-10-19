import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import numpy as np
from pathlib import Path
from skfda.representation.grid import FDataGrid
from transformation.fda.fpca import fpca_with_param, basis_smoothing_hyperparameter_tuning, basis_smoothing_with_lambda
from preprocess.ptbxl_preprocess import get_sr, get_landmarks, align_ecg
from scenario_engineering.privacy_leaking_dataset_creation import get_leak_ratios, get_leak_situations
from privacy_attack.aia import attacks as aia_attacks
from privacy_attack.reidentification import attacks as reid_attacks

data_path = Path("data/validation/privacy/")

def load_set(name, split, ts_file=None):
    return {
        "demo": np.load(data_path / f"{name}_{split}_demographics.npy"),
        "ts": np.load(data_path / f"{ts_file or f'{name}_{split}_ts'}.npy"),
    }

def get_landmarks_with_index(signals, sr):
    """
    get_landmarks, plus the indices of the signals it kept. get_landmarks silently drops signals
    without valid beats, so it is run per signal to keep the rows linked to their demographics.
    """
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
        raise ValueError("No signal had valid landmarks.")
    padded = np.full((len(beats), max(len(b) for b in beats), 3), -1, dtype=int)
    for i, b in enumerate(beats):
        padded[i, :len(b)] = b
    return padded, np.array(kept)

class FPCAFeatures:
    """
    Time series features for the greybox attacks: FPC scores of landmark-aligned ECGs.
    Fit: signals -> get_landmarks -> align_ecg -> smoothing -> FPCA, keeping the FPCA grid points.
    Transform: signals -> get_landmarks -> align_ecg -> smoothing -> projection onto the FPCA grid.
    Signals without valid landmarks get NaN scores (imputed by the attacks).
    """
    def __init__(self, n_components=10, domain_range=(0, 1), sr=100):
        self.n_components = n_components
        self.domain_range = domain_range
        self.sr = sr

    def _align(self, ts):
        landmarks, kept = get_landmarks_with_index(ts, self.sr)
        fd = align_ecg(ts[kept], landmarks)
        if fd.n_samples != len(kept):
            raise ValueError("align_ecg dropped signals; rows would no longer match their demographics.")
        return fd, kept

    def fit(self, ts):
        fd, kept = self._align(ts)
        _, n_timepoints, _ = fd.data_matrix.shape
        self.n_basis = int(n_timepoints / 2)
        lambda_ = basis_smoothing_hyperparameter_tuning(fd, self.n_basis, self.domain_range)
        fd_smooth, _, _, _ = basis_smoothing_with_lambda(fd, lambda_, self.n_basis, self.domain_range)
        _, _, _, self.var_ratio, self.fpca_ = fpca_with_param(fd_smooth, self.n_components)
        self.grid = self.fpca_.components_.grid_points[0]
        self.n_fit, self.n_fit_dropped = len(kept), len(ts) - len(kept)
        return self

    def transform(self, ts):
        fd, kept = self._align(ts)
        lambda_ = basis_smoothing_hyperparameter_tuning(fd, self.n_basis, self.domain_range)
        fd_smooth, _, _, _ = basis_smoothing_with_lambda(fd, lambda_, self.n_basis, self.domain_range)
        fd_smooth = FDataGrid(data_matrix=fd_smooth(self.grid), grid_points=self.grid)

        scores = np.full((len(ts), self.n_components), np.nan)
        scores[kept] = self.fpca_.transform(fd_smooth)
        self.n_dropped = len(ts) - len(kept)
        return scores

# Metrics recorded for the validation. Age uses MAE; sex uses accuracy and AUC, each against its baseline.
aia_metrics = {
    "age": ("mae", "baseline_mae"),
    "sex": ("accuracy", "auc", "baseline_accuracy"),
}
reid_metrics = ("top1_accuracy", "top5_accuracy", "top10_accuracy", "mrr")

def summarize(runs, metric_names):
    return {
        "mean": {k: float(np.mean([r[k] for r in runs])) for k in metric_names},
        "std": {k: float(np.std([r[k] for r in runs])) for k in metric_names},
        "runs": [{k: r[k] for k in metric_names} for r in runs],
    }

def run_attack(attack_cls, synthetic, holdout, target, seeds, metric_names, **attack_kwargs):
    """Fit on the synthetic release plus holdout, then score the whole real training set."""
    runs = []
    for seed in seeds:
        attack = attack_cls(seed=seed, **attack_kwargs)
        attack.fit(synthetic["demo"], synthetic["ts"], synthetic["feat"],
                   holdout["demo"], holdout["ts"], holdout["feat"])
        runs.append(attack.evaluate(target["demo"], target["ts"], target["feat"]))
    return summarize(runs, metric_names)

if __name__ == "__main__":
    save_path = "images/privacy_val/"
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)

    sr = get_sr()
    n_components = 10
    domain_range = (0, 1)
    seeds = [0, 1, 2]
    # Sensitive attribute -> task for the AIA
    aia_targets = {"sex": "classification", "age": "regression"}
    reid_kwargs = {"epochs": 500, "patience": 30}

    np.random.seed(42)
    n_dropped = {}

    ### FPCA fitted on the Holdout set (train + test)
    holdout_ts = np.concatenate([np.load(data_path / f"holdout_{split}_ts.npy") for split in ("train", "test")])
    fpca_features = FPCAFeatures(n_components, domain_range, sr).fit(holdout_ts)
    print(f"FPCA fitted on {fpca_features.n_fit} holdout ECGs ({fpca_features.n_fit_dropped} without landmarks)")
    holdout = {
        "demo": np.concatenate([np.load(data_path / f"holdout_{split}_demographics.npy") for split in ("train", "test")]),
        "ts": holdout_ts,
        "feat": fpca_features.transform(holdout_ts),
    }
    n_dropped["holdout"] = fpca_features.n_dropped
    print(f"Holdout auxiliary set: {len(holdout['demo'])} records")

    ### Real (member) Data: attack targets only
    real = {split: load_set("real", split) for split in ("train", "test")}
    for split, dataset in real.items():
        dataset["feat"] = fpca_features.transform(dataset["ts"])
        n_dropped[f"real_{split}"] = fpca_features.n_dropped

    if not np.array_equal(np.load(data_path / "synthetic_train_index.npy"), np.load(data_path / "real_train_index.npy")):
        raise ValueError("Synthetic train rows are not aligned with their real train source rows.")

    ### One attack run per leak situation and ratio
    # The attacks train on that situation's synthetic train split plus the holdout auxiliary
    # set, and are scored on all real train members. The ratio r is the overall fraction of
    # synthetic rows in the memorized subset.
    situations = get_leak_situations()
    result_tracking = {
        "config": {
            "seeds": seeds,
            "n_components": n_components,
            "fpca_var_ratio": fpca_features.var_ratio.tolist(),
            "fpca_n_fit": fpca_features.n_fit,
            "fpca_n_fit_dropped": fpca_features.n_fit_dropped,
            "n_without_landmarks": n_dropped,
            "aia_targets": aia_targets,
            "reid_kwargs": reid_kwargs,
            "leak_situations": [situation["name"] for situation in situations],
            "leak_ratios": get_leak_ratios(),
            "n_holdout": len(holdout["demo"]),
        },
    }
    for situation in situations:
        result_tracking[situation["name"]] = {}
        for ratio in get_leak_ratios():
            key = f"{ratio:.2f}"
            stem = f"synthetic_{situation['name']}_{key}"
            synthetic = {
                "demo": np.load(data_path / f"{stem}_train_demographics.npy"),
                "ts": np.load(data_path / f"{stem}_train_ts.npy"),
            }
            synthetic["feat"] = fpca_features.transform(synthetic["ts"])
            n_dropped[f"{stem}_train"] = fpca_features.n_dropped
            result_tracking[situation["name"]][key] = {
                "n_train": len(real["train"]["demo"]),
                "aia": {},
                "reidentification": {},
            }

            for sensitive_col, task in aia_targets.items():
                result_tracking[situation["name"]][key]["aia"][sensitive_col] = {}
                for attack_name, attack_cls in aia_attacks.items():
                    result_tracking[situation["name"]][key]["aia"][sensitive_col][attack_name] = run_attack(
                        attack_cls, synthetic, holdout, real["train"], seeds, aia_metrics[sensitive_col],
                        sensitive_col=sensitive_col, task=task, verbose=False)
                    print(f"[{situation['name']} {key}] {attack_name} AIA ({sensitive_col}) done")

            for attack_name, attack_cls in reid_attacks.items():
                result_tracking[situation["name"]][key]["reidentification"][attack_name] = run_attack(
                    attack_cls, synthetic, holdout, real["train"], seeds, reid_metrics, verbose=False, **reid_kwargs)
                print(f"[{situation['name']} {key}] {attack_name} Re-ID done")

            with open(save_path + "privacy_val_result.json", "w") as f:
                json.dump(result_tracking, f, indent=2)

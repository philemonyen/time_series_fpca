import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import pickle
import numpy as np
from pathlib import Path
from transformation.nonlinear.diffusion_map import DenseDiffusionMap
from metrics.fidelity import *
from validation.privacy import *



if __name__ == "__main__":
    results_path = Path("images/sssd_ecg_test")
    results_path.mkdir(parents=True, exist_ok=True)
    result = {}
    domain_range = (0, 1)

    data_path_mia = Path("data/validation/privacy/mia/")
    data_path_reid = Path("data/validation/privacy/reidentification/")

    ### Load SSSD-ECG Synthetic Data
    synthetic_ecg = np.load("data/synthetic_final.npy")
    synthetic_label = np.load("data/synthetic_final_labels.npy")
    norm_indices = np.where(synthetic_label[:, 0] == 1)[0]
    synthetic_norm = synthetic_ecg[norm_indices][:, 1, :].squeeze()

    ### Get Real Unaligned and Aligned Data
    with open(f"data/validation/real_data.pkl", "rb") as f:
        real_data = pickle.load(f)
    with open(f"data/validation/real_fd.pkl", "rb") as f:
        real_fd = pickle.load(f)
    n_sample, n_timepoints, n_channel = real_fd.data_matrix.shape
    n_basis = int(n_timepoints / 2)

    ### ---- Privacy Evaluation ---- ###
    # Audit against SSSD normal synthetics (no leak-ratio sweep).
    seeds = [0, 1, 2]
    sr = 100
    points_per_beat = 100
    lead = 1  # match privacy audit / dataset_creation lead
    mia_kwargs = {"epochs": 100, "batch_size": 64, "patience": 15}
    reid_kwargs = {"epochs": 100, "batch_size": 64, "patience": 15, "margin": 1.0}

    # Privacy cohorts are single-lead (N, 1000); SSSD arrays are (N, 12, 1000).
    synthetic_norm_ts = np.asarray(synthetic_norm, dtype=np.float64)
    if synthetic_norm_ts.ndim == 3:
        synthetic_norm_ts = synthetic_norm_ts[:, lead, :]

    print("Aligning holdout ECGs, then fitting FPCA and DMap...")
    holdout_train_ts = np.load(data_path_reid / "shadow_train_real_ts.npy")
    holdout_test_ts = np.load(data_path_reid / "shadow_test_real_ts.npy")

    holdout_train_aligned, max_beats = align_batch(holdout_train_ts, sr, points_per_beat)
    holdout_test_aligned, _ = align_batch(
        holdout_test_ts, sr, points_per_beat, max_beats=max_beats
    )
    full_holdout_aligned = np.concatenate([holdout_train_aligned, holdout_test_aligned], axis=0)

    fpca = FPCAFeatures(domain_range).fit(full_holdout_aligned)
    holdout_scores = fpca.transform(full_holdout_aligned)

    dmap = DenseDiffusionMap(n_evecs=30, k=20, metric='cosine').fit(holdout_scores)

    def feat_from_unaligned(ts):
        aligned, _ = align_batch(ts, sr, points_per_beat, max_beats=max_beats)
        return fpca.transform(aligned)

    print("Loading Re-Identification Datasets...")
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

    reid_audit_real = {
        "ts": np.load(data_path_reid / "audit_real_ts.npy"),
    }
    reid_audit_real["feat"] = feat_from_unaligned(reid_audit_real["ts"])
    reid_audit_real["dmap"] = dmap.transform(reid_audit_real["feat"])

    print("Loading Membership Inference Datasets...")
    mia_train_ts = np.load(data_path_mia / "shadow_train_ts.npy")
    mia_train_labels = np.load(data_path_mia / "shadow_train_labels.npy")
    mia_train_feat = feat_from_unaligned(mia_train_ts)
    mia_train_dmap = dmap.transform(mia_train_feat)

    mia_val_ts = np.load(data_path_mia / "shadow_test_ts.npy")
    mia_val_labels = np.load(data_path_mia / "shadow_test_labels.npy")
    mia_val_feat = feat_from_unaligned(mia_val_ts)
    mia_val_dmap = dmap.transform(mia_val_feat)

    mia_audit_real_ts = np.load(data_path_mia / "audit_real_ts.npy")
    mia_audit_real_labels = np.load(data_path_mia / "audit_real_labels.npy")
    mia_audit_real_feat = feat_from_unaligned(mia_audit_real_ts)
    mia_audit_real_dmap = dmap.transform(mia_audit_real_feat)

    # Match audit-real count so MIA is balanced and ReID diagonal scoring is valid.
    rng = np.random.default_rng(0)
    n_audit = len(mia_audit_real_ts)
    if len(synthetic_norm_ts) >= n_audit:
        synth_pick = rng.choice(len(synthetic_norm_ts), size=n_audit, replace=False)
    else:
        synth_pick = rng.choice(len(synthetic_norm_ts), size=n_audit, replace=True)
    audit_synth_ts = synthetic_norm_ts[synth_pick]

    reid_audit_synth = {"ts": audit_synth_ts}
    reid_audit_synth["feat"] = feat_from_unaligned(reid_audit_synth["ts"])
    reid_audit_synth["dmap"] = dmap.transform(reid_audit_synth["feat"])

    mia_audit_synth_ts = audit_synth_ts
    mia_audit_synth_labels = np.ones(len(mia_audit_synth_ts))
    mia_audit_synth_feat = feat_from_unaligned(mia_audit_synth_ts)
    mia_audit_synth_dmap = dmap.transform(mia_audit_synth_feat)

    mia_audit_ts = np.concatenate([mia_audit_real_ts, mia_audit_synth_ts], axis=0)
    mia_audit_feat = np.concatenate([mia_audit_real_feat, mia_audit_synth_feat], axis=0)
    mia_audit_dmap = np.concatenate([mia_audit_real_dmap, mia_audit_synth_dmap], axis=0)
    mia_audit_labels = np.concatenate([mia_audit_real_labels, mia_audit_synth_labels], axis=0)

    privacy_result = {"mia": {}, "reidentification": {}}

    print("Evaluating MIA on SSSD synthetic_norm...")
    for attack_name, attack_cls in mia_attacks.items():
        privacy_result["mia"][attack_name] = run_mia_evaluation(
            attack_cls, mia_train_ts, mia_train_feat, mia_train_dmap, mia_train_labels,
            mia_val_ts, mia_val_feat, mia_val_dmap, mia_val_labels,
            mia_audit_ts, mia_audit_feat, mia_audit_dmap, mia_audit_labels,
            seeds, mia_metrics, **mia_kwargs
        )

    print("Evaluating Re-Identification on SSSD synthetic_norm...")
    for attack_name, attack_cls in reid_attacks.items():
        privacy_result["reidentification"][attack_name] = run_reid_evaluation(
            attack_cls, reid_shadow_train, reid_shadow_val, reid_audit_real, reid_audit_synth,
            seeds, reid_metrics, **reid_kwargs
        )

    result["privacy"] = privacy_result
    out_path = results_path / "sssdecg_privacy_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {out_path}")

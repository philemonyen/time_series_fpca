"""
Re-identification attacks on synthetic ECG data.

The adversary trains on the synthetic release plus the holdout records, then links
target records to a patient_id. Every box setting sees the holdout through the
inputs it is allowed to use:

    Tabular Re-ID   : demographics                                          -> patient_id
    Blackbox Re-ID  : demographics + proj(time series)                       -> patient_id
    Greybox Re-ID   : demographics + proj(time series) + time series features -> patient_id

Shapes: demographics N x 5 (patient_id, age, sex, height, weight), time series N x 1000,
time series features N x 10. Raw PTB-XL demographics (N x 7) are also accepted; ecg_id and
strat_fold are dropped. patient_id is the label, so the fused input is 4 (tabular),
4 + 32 (blackbox) or 4 + 32 + 10 (greybox) wide.

Usage:
    python -m privacy_attack.reidentification \
        --synthetic-demo SD.pkl --synthetic-ts ST.npy --synthetic-feat SF.npy \
        --target-demo RD.pkl --target-ts RT.npy --target-feat RF.npy
"""
import argparse

import numpy as np
import pandas as pd
import torch

from privacy_attack.aia import AttributeInferenceAttack, add_common_args, run_attacks

reidentification_columns = ['patient_id', 'age', 'sex', 'height', 'weight']


class ReidentificationAttack(AttributeInferenceAttack):
    """
    Classifies each record into one of the patient_ids seen in training.

    Training rows are the synthetic release plus the holdout records, for every box setting.
    A patient_id can only be predicted if it was seen, so by default the model is fit on all of
    those rows (val_size=0) with a wider head to memorise the id mapping.
    """
    kind = "Re-ID"
    columns = reidentification_columns

    def __init__(self, hidden_dims=(256, 256), dropout=0.1, val_size=0.0, top_k=(1, 5, 10), **kwargs):
        super().__init__(sensitive_col="patient_id", task="classification", hidden_dims=hidden_dims,
                         dropout=dropout, val_size=val_size, **kwargs)
        self.top_k = top_k

    def rank(self, demo, ts=None, feat=None):
        """Rank (1 = best) of each target's true patient_id; inf when the id is not in the synthetic data."""
        logits, y_raw = self._forward(demo, ts, feat)
        y = self._encode_target(y_raw, fit=False)
        valid = pd.Series(y_raw).notna().to_numpy()
        logits, y = torch.as_tensor(logits[valid]), torch.as_tensor(y[valid])

        ranks = torch.full((len(y),), float("inf"))
        seen = y >= 0
        true_score = logits[seen].gather(1, y[seen].unsqueeze(1))
        ranks[seen] = ((logits[seen] > true_score).sum(dim=1) + 1).float()
        return ranks.numpy()

    def evaluate(self, demo, ts=None, feat=None):
        """Linkage performance on target records whose true patient_id is known."""
        ranks = self.rank(demo, ts, feat)
        seen = np.isfinite(ranks)
        n_ids = len(self.classes_)

        metrics = {f"top{k}_accuracy": np.mean(ranks <= k) for k in self.top_k}
        metrics["mrr"] = np.mean(np.where(seen, 1.0 / ranks, 0.0))
        metrics["coverage"] = np.mean(seen)
        metrics["top1_accuracy_covered"] = np.mean(ranks[seen] <= 1) if seen.any() else float("nan")
        metrics["n_targets"] = len(ranks)
        metrics["n_synthetic_ids"] = n_ids
        return {k: float(v) for k, v in metrics.items()}


class TabularReID(ReidentificationAttack):
    """Demographics from the synthetic release and the holdout set."""
    name = "tabular"


class BlackboxReID(ReidentificationAttack):
    """Demographics plus the projected time series, from the synthetic release and the holdout set."""
    name = "blackbox"
    uses_ts = True


class GreyboxReID(ReidentificationAttack):
    """Demographics, projected time series, and time series features, from the synthetic release and the holdout set."""
    name = "greybox"
    uses_ts = True
    uses_feat = True


attacks = {"tabular": TabularReID, "blackbox": BlackboxReID, "greybox": GreyboxReID}


def parse_args():
    parser = argparse.ArgumentParser(description="Re-identification attacks on synthetic ECG data.")
    add_common_args(parser, list(attacks), "results/reidentification_results.json")
    return parser.parse_args()


if __name__ == "__main__":
    run_attacks(parse_args(), attacks)

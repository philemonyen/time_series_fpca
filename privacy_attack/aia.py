"""
Attribute Inference Attacks (AIA) on synthetic ECG data.

The adversary trains on the synthetic release and infers a sensitive attribute
of target (real) records from the attributes it already knows:

    Tabular AIA   : demographics (minus sensitive attr)                    -> sensitive attr
    Blackbox AIA  : demographics + proj(time series)                       -> sensitive attr
    Greybox AIA   : demographics + proj(time series) + time series features -> sensitive attr

Shapes: demographics N x 4 (age, sex, height, weight), time series N x 1000,
time series features N x 10. Raw PTB-XL demographics (N x 7) are also accepted;
the identifier columns (ecg_id, patient_id, strat_fold) are dropped.
The time series is projected to `embed_dim` (default 32) before fusion, so the
fused input is 3 (tabular), 3 + 32 (blackbox) or 3 + 32 + 10 (greybox) wide.

Usage:
    python -m privacy_attack.aia \
        --synthetic-demo SD.pkl --synthetic-ts ST.npy --synthetic-feat SF.npy \
        --target-demo RD.pkl --target-ts RT.npy --target-feat RF.npy \
        --sensitive-col sex --task classification
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ptbxl_demographic_columns = ['ecg_id', 'patient_id', 'age', 'sex', 'height', 'weight', 'strat_fold']
demographic_columns = ['age', 'sex', 'height', 'weight']


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#### ---- Models ---- ####
class TimeSeriesProjection(nn.Module):
    """Projects a raw time series (N x ts_len) to a low-dimensional embedding (N x embed_dim)."""
    def __init__(self, ts_len=1000, embed_dim=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ts_len, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, ts):
        return self.proj(ts)


class AIANetwork(nn.Module):
    """
    MLP over the concatenation of [demographics, proj(time series), time series features].
    The time series / feature branches are only built when their dimensions are given.
    """
    def __init__(self, demo_dim, out_dim, ts_len=None, feat_dim=None, embed_dim=32,
                 hidden_dims=(64, 32), dropout=0.2):
        super().__init__()
        self.ts_proj = TimeSeriesProjection(ts_len, embed_dim) if ts_len else None
        in_dim = demo_dim + (embed_dim if ts_len else 0) + (feat_dim or 0)

        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, demo, ts=None, feat=None):
        parts = [demo]
        if self.ts_proj is not None:
            parts.append(self.ts_proj(ts))
        if feat is not None:
            parts.append(feat)
        return self.head(torch.cat(parts, dim=1))


#### ---- Preprocessing ---- ####
class _Standardizer:
    """Median imputation + z-scoring, fitted on the synthetic (training) data only."""
    def __init__(self, scalar=False):
        self.scalar = scalar

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        axis = None if self.scalar else 0
        self.median = np.nanmedian(x, axis=axis)
        x = np.where(np.isnan(x), self.median, x)
        self.mean = x.mean(axis=axis)
        self.std = x.std(axis=axis)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        x = np.where(np.isnan(x), self.median, x)
        return ((x - self.mean) / self.std).astype(np.float32)


def _to_frame(demo, columns=demographic_columns):
    """Demographics as a DataFrame restricted to `columns`."""
    if not isinstance(demo, pd.DataFrame):
        demo = np.asarray(demo, dtype=object)
        if demo.shape[1] == len(ptbxl_demographic_columns):
            demo = pd.DataFrame(demo, columns=ptbxl_demographic_columns)
        elif demo.shape[1] == len(columns):
            demo = pd.DataFrame(demo, columns=columns)
        else:
            raise ValueError(f"Expected demographics with columns {columns}, got shape {demo.shape}.")
    missing = [c for c in columns if c not in demo.columns]
    if missing:
        raise ValueError(f"Demographics are missing columns: {missing}")
    return demo[columns].reset_index(drop=True)


def _numeric(df):
    return df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


#### ---- Attacks ---- ####
class AttributeInferenceAttack:
    """
    Base AIA. Subclasses set which modalities are used.

    sensitive_col: one of `columns` (or its index), removed from the inputs.
    task: "classification" (categorical attr, e.g. sex) or "regression" (continuous attr, e.g. age).
    val_size: fraction of the synthetic data held out for early stopping; 0 trains on all of it
              and early-stops on the training loss.
    """
    uses_ts = False
    uses_feat = False
    name = "base"
    kind = "AIA"
    columns = demographic_columns

    def __init__(self, sensitive_col, task="classification", embed_dim=32,
                 hidden_dims=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4, batch_size=256,
                 epochs=200, patience=20, val_size=0.2, seed=0, device=None, verbose=True):
        if task not in ("classification", "regression"):
            raise ValueError(f"Unknown task: {task}")
        self.sensitive_col = sensitive_col
        self.task = task
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_size = val_size
        self.seed = seed
        self.device = device or get_device()
        self.verbose = verbose

    # -- input handling --
    def _split_demo(self, demo):
        df = _to_frame(demo, self.columns)
        target = self.sensitive_col
        if isinstance(target, int):
            target = self.columns[target]
        if target not in self.columns:
            raise ValueError(f"sensitive_col must be one of {self.columns}, got {target!r}.")
        return df.drop(columns=[target]), df[target]

    def _check_inputs(self, ts, feat):
        if self.uses_ts and ts is None:
            raise ValueError(f"{self.name} {self.kind} requires the time series.")
        if self.uses_feat and feat is None:
            raise ValueError(f"{self.name} {self.kind} requires the time series features.")

    def _prepare(self, demo, ts, feat, fit):
        self._check_inputs(ts, feat)
        known, y = self._split_demo(demo)
        if fit:
            self.input_columns = list(known.columns)
            self.demo_scaler = _Standardizer().fit(_numeric(known))
            if self.uses_ts:
                self.ts_scaler = _Standardizer(scalar=True).fit(np.asarray(ts).reshape(len(ts), -1))
            if self.uses_feat:
                self.feat_scaler = _Standardizer().fit(np.asarray(feat).reshape(len(feat), -1))
        else:
            known = known[self.input_columns]

        inputs = {"demo": self.demo_scaler.transform(_numeric(known))}
        if self.uses_ts:
            inputs["ts"] = self.ts_scaler.transform(np.asarray(ts).reshape(len(ts), -1))
        if self.uses_feat:
            inputs["feat"] = self.feat_scaler.transform(np.asarray(feat).reshape(len(feat), -1))
        return inputs, y

    def _encode_target(self, y, fit):
        if self.task == "classification":
            if fit:
                self.classes_ = np.array(sorted(pd.Series(y).dropna().unique(), key=str))
            lookup = {c: i for i, c in enumerate(self.classes_)}
            # Labels unseen in the synthetic data map to -1 and can never be predicted correctly.
            return np.array([lookup.get(v, -1) for v in y], dtype=np.int64)
        y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=np.float64)
        if fit:
            self.y_mean = np.nanmean(y)
            self.y_std = np.nanstd(y) or 1.0
        return ((y - self.y_mean) / self.y_std).astype(np.float32)

    def _to_tensors(self, inputs, idx=None):
        return {k: torch.as_tensor(v if idx is None else v[idx], device=self.device) for k, v in inputs.items()}

    # -- training --
    def fit(self, demo, ts=None, feat=None):
        """Train the attack model on the synthetic release."""
        set_seed(self.seed)
        inputs, y_raw = self._prepare(demo, ts, feat, fit=True)
        y = self._encode_target(y_raw, fit=True)
        valid = (y >= 0) if self.task == "classification" else ~np.isnan(y)
        idx = np.where(valid)[0]
        if self.task == "classification":
            self.majority_class_ = np.bincount(y[idx]).argmax()

        if self.val_size > 0:
            stratify = y[idx] if self.task == "classification" and np.bincount(y[idx]).min() >= 2 else None
            train_idx, val_idx = train_test_split(idx, test_size=self.val_size, random_state=self.seed,
                                                  stratify=stratify)
        else:
            train_idx = val_idx = idx

        out_dim = len(self.classes_) if self.task == "classification" else 1
        self.model = AIANetwork(
            demo_dim=inputs["demo"].shape[1],
            out_dim=out_dim,
            ts_len=inputs["ts"].shape[1] if self.uses_ts else None,
            feat_dim=inputs["feat"].shape[1] if self.uses_feat else None,
            embed_dim=self.embed_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)
        loss_fn = nn.CrossEntropyLoss() if self.task == "classification" else nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        x_train, x_val = self._to_tensors(inputs, train_idx), self._to_tensors(inputs, val_idx)
        y_train = torch.as_tensor(y[train_idx], device=self.device)
        y_val = torch.as_tensor(y[val_idx], device=self.device)
        if self.task == "regression":
            y_train, y_val = y_train.unsqueeze(1), y_val.unsqueeze(1)

        best_loss, best_state, wait = np.inf, None, 0
        n_train = len(train_idx)
        for epoch in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n_train, device=self.device)
            for start in range(0, n_train, self.batch_size):
                batch = perm[start:start + self.batch_size]
                if len(batch) < 2:  # BatchNorm needs > 1 sample
                    continue
                optimizer.zero_grad()
                out = self.model(**{k: v[batch] for k, v in x_train.items()})
                loss = loss_fn(out, y_train[batch])
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model(**x_val), y_val).item()
            if val_loss < best_loss - 1e-6:
                best_loss, wait = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        self.best_val_loss = best_loss
        if self.verbose:
            print(f"[{self.name} {self.kind}] stopped at epoch {epoch + 1}, best val loss {best_loss:.4f}")
        return self

    # -- inference --
    @torch.no_grad()
    def _forward(self, demo, ts, feat):
        inputs, y_raw = self._prepare(demo, ts, feat, fit=False)
        self.model.eval()
        x = self._to_tensors(inputs)
        n = len(inputs["demo"])
        outs = [self.model(**{k: v[i:i + 4096] for k, v in x.items()}) for i in range(0, n, 4096)]
        return torch.cat(outs).cpu().numpy(), y_raw

    def predict_proba(self, demo, ts=None, feat=None):
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")
        logits, _ = self._forward(demo, ts, feat)
        return torch.softmax(torch.as_tensor(logits), dim=1).numpy()

    def predict(self, demo, ts=None, feat=None):
        """Infer the sensitive attribute of target records (their sensitive column is ignored)."""
        out, _ = self._forward(demo, ts, feat)
        if self.task == "classification":
            return self.classes_[out.argmax(axis=1)]
        return out[:, 0] * self.y_std + self.y_mean

    def evaluate(self, demo, ts=None, feat=None):
        """Attack performance on target records whose true sensitive attribute is known."""
        out, y_raw = self._forward(demo, ts, feat)
        y = self._encode_target(y_raw, fit=False)

        if self.task == "classification":
            valid = pd.Series(y_raw).notna().to_numpy()
            y, out = y[valid], out[valid]
            proba = torch.softmax(torch.as_tensor(out), dim=1).numpy()
            pred = proba.argmax(axis=1)
            metrics = {
                "accuracy": accuracy_score(y, pred),
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro"),
                "baseline_accuracy": accuracy_score(y, np.full_like(y, self.majority_class_)),
            }
            seen = y >= 0
            try:
                if len(self.classes_) == 2:
                    metrics["auc"] = roc_auc_score(y[seen], proba[seen, 1])
                else:
                    metrics["auc"] = roc_auc_score(y[seen], proba[seen], multi_class="ovr",
                                                   labels=np.arange(len(self.classes_)))
            except ValueError:
                metrics["auc"] = float("nan")
        else:
            valid = ~np.isnan(y)
            y_true = y[valid] * self.y_std + self.y_mean
            y_pred = out[valid, 0] * self.y_std + self.y_mean
            metrics = {
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": r2_score(y_true, y_pred),
                "baseline_mae": mean_absolute_error(y_true, np.full_like(y_true, self.y_mean)),
            }
        metrics["n_targets"] = int(valid.sum())
        return {k: float(v) for k, v in metrics.items()}

    def fit_evaluate(self, synthetic, target):
        """synthetic / target: dicts with keys "demo", "ts", "feat"."""
        self.fit(synthetic["demo"], synthetic.get("ts"), synthetic.get("feat"))
        return self.evaluate(target["demo"], target.get("ts"), target.get("feat"))


class TabularAIA(AttributeInferenceAttack):
    """Synthetic demographics only."""
    name = "tabular"


class BlackboxAIA(AttributeInferenceAttack):
    """Synthetic demographics + projected synthetic time series."""
    name = "blackbox"
    uses_ts = True


class GreyboxAIA(AttributeInferenceAttack):
    """Synthetic demographics + projected synthetic time series + time series features."""
    name = "greybox"
    uses_ts = True
    uses_feat = True


attacks = {"tabular": TabularAIA, "blackbox": BlackboxAIA, "greybox": GreyboxAIA}


#### ---- CLI ---- ####
def load_array(path):
    if path is None:
        return None
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj if isinstance(obj, (pd.DataFrame, np.ndarray)) else np.asarray(obj, dtype=object)
    raise ValueError(f"Unsupported file type: {path}")


def add_common_args(parser, attack_names, default_output):
    parser.add_argument("--synthetic-demo", required=True)
    parser.add_argument("--synthetic-ts")
    parser.add_argument("--synthetic-feat")
    parser.add_argument("--target-demo", required=True)
    parser.add_argument("--target-ts")
    parser.add_argument("--target-feat")
    parser.add_argument("--attacks", nargs="+", choices=attack_names, default=attack_names)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output", default=default_output)


def run_attacks(args, attack_classes, **attack_kwargs):
    """Fit/evaluate each selected attack over all seeds and save the results as JSON."""
    synthetic = {"demo": load_array(args.synthetic_demo), "ts": load_array(args.synthetic_ts),
                 "feat": load_array(args.synthetic_feat)}
    target = {"demo": load_array(args.target_demo), "ts": load_array(args.target_ts),
              "feat": load_array(args.target_feat)}
    print(f"Device: {get_device()}")

    results = {}
    for attack_name in args.attacks:
        runs = []
        for seed in args.seeds:
            attack = attack_classes[attack_name](
                embed_dim=args.embed_dim, epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size, lr=args.lr, seed=seed, **attack_kwargs,
            )
            runs.append(attack.fit_evaluate(synthetic, target))
            print(f"[{attack_name} {attack.kind}] seed {seed}: {runs[-1]}")
        results[attack_name] = {
            "runs": runs,
            "mean": {k: float(np.mean([r[k] for r in runs])) for k in runs[0]},
            "std": {k: float(np.std([r[k] for r in runs])) for k in runs[0]},
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"Saved results to {output}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Attribute inference attacks on synthetic ECG data.")
    add_common_args(parser, list(attacks), "results/aia_results.json")
    parser.add_argument("--sensitive-col", required=True, choices=demographic_columns,
                        help="Sensitive attribute to infer.")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_attacks(args, attacks, sensitive_col=args.sensitive_col, task=args.task)

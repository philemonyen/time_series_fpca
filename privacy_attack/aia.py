"""
Attribute Inference Attacks (AIA) on synthetic ECG data.

The adversary trains on the synthetic training split plus the holdout (auxiliary
real) records, after those rows are shuffled together. Performance is reported
on the synthetic test split. The adversary is not scored on the real members.
Every box setting sees the holdout through the inputs it is allowed to use:

    Tabular AIA   : demographics (minus sensitive attr)                    -> sensitive attr
    Blackbox AIA  : demographics + proj(time series)                       -> sensitive attr
    Greybox AIA   : demographics + proj(time series) + time series features -> sensitive attr

Shapes: demographics N x 4 (age, sex, height, weight), time series N x L
(the shared landmark-aligned length), time series features N x 10. Raw PTB-XL
demographics (N x 7) are also accepted;
the identifier columns (ecg_id, patient_id, strat_fold) are dropped.
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
class AIANetwork(nn.Module):
    """
    Late-Fusion architecture to capture non-linear correlations between ECG modalities
    and target attributes independently before fusion.
    """
    def __init__(self, demo_dim, out_dim, ts_len=None, feat_dim=None, embed_dim=32,
                 hidden_dims=(128, 64, 32), dropout=0.3):
        super().__init__()
        self.uses_ts = ts_len is not None
        self.uses_feat = feat_dim is not None

        # 1. Independent Demographic Branch
        self.demo_branch = nn.Sequential(
            nn.Linear(demo_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. Independent TS Branch (if applicable)
        if self.uses_ts:
            self.ts_proj = nn.Sequential(
                nn.Linear(ts_len, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # 3. Independent Feat Branch (if applicable)
        if self.uses_feat:
            self.feat_branch = nn.Sequential(
                nn.Linear(feat_dim, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # 4. Fusion Head
        fused_dim = 32 + (embed_dim if self.uses_ts else 0) + (16 if self.uses_feat else 0)
        
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(fused_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            fused_dim = h
        layers.append(nn.Linear(fused_dim, out_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, demo, ts=None, feat=None):
        # Explicitly cast to float32
        demo = demo.float()
        d_emb = self.demo_branch(demo)
        parts = [d_emb]
        
        if self.uses_ts and ts is not None:
            ts = ts.float()
            parts.append(self.ts_proj(ts))
        
        if self.uses_feat and feat is not None:
            feat = feat.float()
            parts.append(self.feat_branch(feat))
            
        return self.head(torch.cat(parts, dim=1))

#### ---- Preprocessing ---- ####
class _Standardizer:
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
    uses_ts = False
    uses_feat = False
    name = "base"
    kind = "AIA"
    columns = demographic_columns

    def __init__(self, sensitive_col, task="classification", embed_dim=32,
                 hidden_dims=(128, 64, 32), dropout=0.3, lr=1e-3, weight_decay=1e-4, batch_size=256,
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
            raise ValueError(f"{self.name} requires time series.")
        if self.uses_feat and feat is None:
            raise ValueError(f"{self.name} requires features.")

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
            return np.array([lookup.get(v, -1) for v in y], dtype=np.int64)
        y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=np.float64)
        if fit:
            self.y_mean = np.nanmean(y)
            self.y_std = np.nanstd(y) or 1.0
        return ((y - self.y_mean) / self.y_std).astype(np.float32)

    def _to_tensors(self, inputs, idx=None):
        return {k: torch.as_tensor(v if idx is None else v[idx], device=self.device) for k, v in inputs.items()}

    def _concat_demo(self, demo, extra):
        if isinstance(demo, pd.DataFrame) or isinstance(extra, pd.DataFrame):
            frames = []
            for part in (demo, extra):
                frames.append(part.reset_index(drop=True) if isinstance(part, pd.DataFrame) else pd.DataFrame(part))
            return pd.concat(frames, ignore_index=True)
        demo, extra = np.asarray(demo), np.asarray(extra)
        return np.concatenate([demo, extra], axis=0)

    def _with_holdout(self, demo, ts, feat, holdout_demo, holdout_ts, holdout_feat):
        if holdout_demo is None:
            return demo, ts, feat
        demo = self._concat_demo(demo, holdout_demo)
        if self.uses_ts:
            ts = np.concatenate([np.asarray(ts), np.asarray(holdout_ts)], axis=0)
        if self.uses_feat:
            feat = np.concatenate([np.asarray(feat), np.asarray(holdout_feat)], axis=0)
        return demo, ts, feat

    def _shuffle_rows(self, demo, ts, feat):
        n = len(demo)
        perm = np.random.default_rng(self.seed).permutation(n)
        if isinstance(demo, pd.DataFrame):
            demo = demo.iloc[perm].reset_index(drop=True)
        else:
            demo = np.asarray(demo)[perm]
        if self.uses_ts: ts = np.asarray(ts)[perm]
        if self.uses_feat: feat = np.asarray(feat)[perm]
        return demo, ts, feat

    def fit(self, demo, ts=None, feat=None, holdout_demo=None, holdout_ts=None, holdout_feat=None):
        set_seed(self.seed)
        demo, ts, feat = self._with_holdout(demo, ts, feat, holdout_demo, holdout_ts, holdout_feat)
        demo, ts, feat = self._shuffle_rows(demo, ts, feat)
        inputs, y_raw = self._prepare(demo, ts, feat, fit=True)
        y = self._encode_target(y_raw, fit=True)
        valid = (y >= 0) if self.task == "classification" else ~np.isnan(y)
        idx = np.where(valid)[0]
        if self.task == "classification":
            self.majority_class_ = np.bincount(y[idx]).argmax()

        if self.val_size > 0:
            stratify = y[idx] if self.task == "classification" and np.bincount(y[idx]).min() >= 2 else None
            train_idx, val_idx = train_test_split(idx, test_size=self.val_size, random_state=self.seed, stratify=stratify)
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
                if len(batch) < 2: continue
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
                if wait >= self.patience: break

        self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _forward(self, demo, ts, feat):
        inputs, y_raw = self._prepare(demo, ts, feat, fit=False)
        self.model.eval()
        x = self._to_tensors(inputs)
        n = len(inputs["demo"])
        outs = [self.model(**{k: v[i:i + 4096] for k, v in x.items()}) for i in range(0, n, 4096)]
        return torch.cat(outs).cpu().numpy(), y_raw

    def evaluate(self, demo, ts=None, feat=None):
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
                if len(self.classes_) == 2: metrics["auc"] = roc_auc_score(y[seen], proba[seen, 1])
                else: metrics["auc"] = roc_auc_score(y[seen], proba[seen], multi_class="ovr", labels=np.arange(len(self.classes_)))
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

    def fit_evaluate(self, synthetic, target, holdout=None):
        holdout = holdout or {}
        self.fit(synthetic["demo"], synthetic.get("ts"), synthetic.get("feat"),
                 holdout.get("demo"), holdout.get("ts"), holdout.get("feat"))
        return self.evaluate(target["demo"], target.get("ts"), target.get("feat"))

class TabularAIA(AttributeInferenceAttack): name = "tabular"
class BlackboxAIA(AttributeInferenceAttack): name = "blackbox"; uses_ts = True
class GreyboxAIA(AttributeInferenceAttack): name = "greybox"; uses_ts = True; uses_feat = True

attacks = {"tabular": TabularAIA, "blackbox": BlackboxAIA, "greybox": GreyboxAIA}

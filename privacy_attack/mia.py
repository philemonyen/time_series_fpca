"""
Membership Inference Attack (MIA) on Time-Series Data.

End-to-end network:
  - TS branch: MLP compressing the raw series (e.g. 1000 -> 32)
  - FPC / DMap branches: single linear projections (e.g. 74 -> 32, 30 -> 32)
  - Active embeddings are concatenated and classified by an MLP head

All parameters share one BCE gradient path.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class MIAEmbedding(nn.Module):
    def __init__(
        self,
        ts_dim,
        feat_dim,
        dmap_dim,
        hidden_dims=(256, 128),
        proj_dim=32,
        dropout=0.1,
    ):
        super().__init__()

        # Blackbox / shared: compress high-dim time series.
        self.ts_net = None
        if ts_dim > 0:
            self.ts_net = nn.Sequential(
                nn.Linear(ts_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, proj_dim),
                nn.ReLU(),
            )

        # Greybox inputs are already informative; use a single projection layer.
        self.feat_net = nn.Linear(feat_dim, proj_dim) if feat_dim > 0 else None
        self.dmap_net = nn.Linear(dmap_dim, proj_dim) if dmap_dim > 0 else None

        in_dim = (
            (proj_dim if ts_dim > 0 else 0)
            + (proj_dim if feat_dim > 0 else 0)
            + (proj_dim if dmap_dim > 0 else 0)
        )

        # Membership classification MLP over concatenated embeddings.
        layers = []
        curr = in_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(curr, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            curr = h
        layers.append(nn.Linear(curr, 1))
        self.classifier = nn.Sequential(*layers)

    def encode(self, ts=None, feat=None, dmap=None):
        parts = []
        if self.ts_net is not None and ts is not None and ts.shape[-1] > 0:
            parts.append(self.ts_net(ts.float()))
        if self.feat_net is not None and feat is not None and feat.shape[-1] > 0:
            parts.append(torch.relu(self.feat_net(feat.float())))
        if self.dmap_net is not None and dmap is not None and dmap.shape[-1] > 0:
            parts.append(torch.relu(self.dmap_net(dmap.float())))
        if not parts:
            raise ValueError("No active modalities provided for encoding.")
        return torch.cat(parts, dim=-1)

    def forward(self, ts=None, feat=None, dmap=None):
        return self.classifier(self.encode(ts, feat, dmap)).squeeze(-1)


class MIADataset(Dataset):
    def __init__(self, ts, feat, dmap, labels):
        self.ts = ts if ts is not None else np.empty((len(labels), 0), dtype=np.float32)
        self.feat = feat if feat is not None else np.empty((len(labels), 0), dtype=np.float32)
        self.dmap = dmap if dmap is not None else np.empty((len(labels), 0), dtype=np.float32)
        self.labels = labels
        self.n = len(labels)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.ts[idx], self.feat[idx], self.dmap[idx], self.labels[idx]


class MIABase:
    kind = "MIA"
    uses_ts = False
    uses_feat = False
    uses_dmap = False

    def __init__(
        self,
        hidden_dims=(256, 128),
        dropout=0.1,
        epochs=100,
        batch_size=64,
        lr=1e-3,
        patience=15,
        seed=42,
        random_state=None,
        **kwargs,
    ):
        self.hidden_dims, self.dropout = hidden_dims, dropout
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.patience = patience
        self.seed = seed if random_state is None else random_state
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def fit(
        self,
        train_ts,
        train_feat,
        train_dmap,
        train_labels,
        val_ts=None,
        val_feat=None,
        val_dmap=None,
        val_labels=None,
    ):
        ts_dim = train_ts.shape[-1] if self.uses_ts and train_ts is not None else 0
        feat_dim = train_feat.shape[-1] if self.uses_feat and train_feat is not None else 0
        dmap_dim = train_dmap.shape[-1] if self.uses_dmap and train_dmap is not None else 0

        self.model = MIAEmbedding(
            ts_dim,
            feat_dim,
            dmap_dim,
            hidden_dims=self.hidden_dims,
            proj_dim=32,
            dropout=self.dropout,
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(
            MIADataset(
                train_ts if self.uses_ts else None,
                train_feat if self.uses_feat else None,
                train_dmap if self.uses_dmap else None,
                train_labels,
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        has_val = val_labels is not None
        if has_val:
            val_loader = DataLoader(
                MIADataset(
                    val_ts if self.uses_ts else None,
                    val_feat if self.uses_feat else None,
                    val_dmap if self.uses_dmap else None,
                    val_labels,
                ),
                batch_size=self.batch_size,
                shuffle=False,
            )

        best_loss, best_state, no_improve = float("inf"), None, 0
        for _ in range(self.epochs):
            self.model.train()
            for t, f, d, l in train_loader:
                logits = self.model(t, f, d)
                loss = criterion(logits, l.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if has_val:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for t, f, d, l in val_loader:
                        logits = self.model(t, f, d)
                        loss = criterion(logits, l.float())
                        val_loss += loss.item() * len(l)
                val_loss /= len(val_loader.dataset)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def evaluate(self, audit_ts, audit_feat, audit_dmap, audit_labels):
        self.model.eval()
        n = len(audit_labels)
        with torch.no_grad():
            t_t = (
                torch.as_tensor(audit_ts, dtype=torch.float32)
                if self.uses_ts and audit_ts is not None and audit_ts.shape[-1] > 0
                else torch.empty((n, 0))
            )
            f_t = (
                torch.as_tensor(audit_feat, dtype=torch.float32)
                if self.uses_feat and audit_feat is not None and audit_feat.shape[-1] > 0
                else torch.empty((n, 0))
            )
            d_t = (
                torch.as_tensor(audit_dmap, dtype=torch.float32)
                if self.uses_dmap and audit_dmap is not None and audit_dmap.shape[-1] > 0
                else torch.empty((n, 0))
            )
            logits = self.model(t_t, f_t, d_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

        audit_labels = np.asarray(audit_labels)
        metrics = {
            "accuracy": float(accuracy_score(audit_labels, preds)),
            "precision": float(precision_score(audit_labels, preds, zero_division=0)),
            "recall": float(recall_score(audit_labels, preds, zero_division=0)),
            "f1": float(f1_score(audit_labels, preds, zero_division=0)),
        }
        if len(np.unique(audit_labels)) > 1:
            metrics["auc"] = float(roc_auc_score(audit_labels, probs))
        else:
            metrics["auc"] = 0.5
        return metrics


class BlackboxMIA(MIABase):
    name = "blackbox"
    uses_ts = True


class GreyboxIMIA(MIABase):
    name = "greybox1"
    uses_ts = True
    uses_feat = True


class GreyboxIIMIA(MIABase):
    name = "greybox2"
    uses_ts = True
    uses_feat = True
    uses_dmap = True


attacks = {"blackbox": BlackboxMIA, "greybox1": GreyboxIMIA, "greybox2": GreyboxIIMIA}

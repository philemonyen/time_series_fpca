"""
Late Fusion Relation Network for Re-Identification Attacks.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class SiameseEmbedding(nn.Module):
    def __init__(self, ts_dim, feat_dim, dmap_dim, hidden_dims=(256, 256), ts_out=64, dropout=0.1):
        super().__init__()
        
        def build_branch(in_d, hiddens, out_d):
            if in_d <= 0: return None
            layers = []
            curr_d = in_d
            for h in hiddens:
                layers.append(nn.Linear(curr_d, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                curr_d = h
            layers.append(nn.Linear(curr_d, out_d))
            return nn.Sequential(*layers)

        self.ts_out = ts_out if ts_dim > 0 else 0
        self.feat_out = min(feat_dim, 10) if feat_dim > 0 else 0
        self.dmap_out = min(dmap_dim, 30) if dmap_dim > 0 else 0

        # Separate independent branches to prevent gradient interference
        self.ts_net = build_branch(ts_dim, hidden_dims, self.ts_out)
        self.feat_net = build_branch(feat_dim, (16, 16), self.feat_out)
        self.dmap_net = build_branch(dmap_dim, (32, 32), self.dmap_out)

    def forward(self, ts=None, feat=None, dmap=None):
        out = {}
        if self.ts_net is not None and ts is not None and ts.shape[-1] > 0:
            out['ts'] = self.ts_net(ts.float())
        if self.feat_net is not None and feat is not None and feat.shape[-1] > 0:
            out['feat'] = self.feat_net(feat.float())
        if self.dmap_net is not None and dmap is not None and dmap.shape[-1] > 0:
            out['dmap'] = self.dmap_net(dmap.float())
        return out

class RelationHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, diff_concat):
        return self.net(diff_concat).squeeze(-1)

class PairDataset(Dataset):
    def __init__(self, real_ts, real_feat, real_dmap, synth_ts, synth_feat, synth_dmap, pos_ts=None, pos_feat=None, pos_dmap=None):
        self.n = len(real_ts) if real_ts is not None else len(real_feat)
        
        self.real_ts = real_ts if real_ts is not None else np.empty((self.n, 0), dtype=np.float32)
        self.real_feat = real_feat if real_feat is not None else np.empty((self.n, 0), dtype=np.float32)
        self.real_dmap = real_dmap if real_dmap is not None else np.empty((self.n, 0), dtype=np.float32)
        
        sn = len(synth_ts) if synth_ts is not None else len(synth_feat)
        self.synth_ts = synth_ts if synth_ts is not None else np.empty((sn, 0), dtype=np.float32)
        self.synth_feat = synth_feat if synth_feat is not None else np.empty((sn, 0), dtype=np.float32)
        self.synth_dmap = synth_dmap if synth_dmap is not None else np.empty((sn, 0), dtype=np.float32)
        
        self.pos_ts = pos_ts if pos_ts is not None else self.real_ts
        self.pos_feat = pos_feat if pos_feat is not None else self.real_feat
        self.pos_dmap = pos_dmap if pos_dmap is not None else self.real_dmap

    def __len__(self): 
        return self.n * 2

    def __getitem__(self, idx):
        real_idx = idx // 2
        is_pos = (idx % 2 == 0)
        
        if is_pos:
            synth_idx = real_idx
            r = (self.real_ts[real_idx], self.real_feat[real_idx], self.real_dmap[real_idx])
            s = (self.pos_ts[synth_idx], self.pos_feat[synth_idx], self.pos_dmap[synth_idx])
            label = 1.0
        else:
            synth_idx = np.random.randint(0, self.n - 1)
            if synth_idx >= real_idx: 
                synth_idx += 1
            r = (self.real_ts[real_idx], self.real_feat[real_idx], self.real_dmap[real_idx])
            s = (self.synth_ts[synth_idx], self.synth_feat[synth_idx], self.synth_dmap[synth_idx])
            label = 0.0
            
        return r, s, label

class SiameseReID:
    kind = "Re-ID"
    uses_ts = False
    uses_feat = False
    uses_dmap = False
    
    def __init__(self, hidden_dims=(256, 256), dropout=0.1, top_k=(1, 5, 10), epochs=100,
                 batch_size=64, lr=1e-3, patience=15, seed=42, **kwargs):
        self.hidden_dims, self.dropout, self.top_k = hidden_dims, dropout, top_k
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.patience, self.seed = patience, seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def fit(self, train_real_ts, train_real_feat, train_real_dmap,
                  train_synth_ts, train_synth_feat, train_synth_dmap,
                  train_pos_ts=None, train_pos_feat=None, train_pos_dmap=None,
                  val_real_ts=None, val_real_feat=None, val_real_dmap=None,
                  val_synth_ts=None, val_synth_feat=None, val_synth_dmap=None,
                  val_pos_ts=None, val_pos_feat=None, val_pos_dmap=None):
        
        ts_dim = train_real_ts.shape[-1] if self.uses_ts and train_real_ts is not None else 0
        feat_dim = train_real_feat.shape[-1] if self.uses_feat and train_real_feat is not None else 0
        dmap_dim = train_real_dmap.shape[-1] if self.uses_dmap and train_real_dmap is not None else 0

        self.embedding = SiameseEmbedding(ts_dim, feat_dim, dmap_dim, self.hidden_dims, ts_out=64, dropout=self.dropout)
        
        # Calculate total dimension for Late Fusion
        head_in_dim = self.embedding.ts_out + self.embedding.feat_out + self.embedding.dmap_out
        self.head = RelationHead(in_dim=head_in_dim)
        
        optimizer = optim.Adam(list(self.embedding.parameters()) + list(self.head.parameters()), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(
            PairDataset(train_real_ts if self.uses_ts else None, train_real_feat if self.uses_feat else None, train_real_dmap if self.uses_dmap else None,
                        train_synth_ts if self.uses_ts else None, train_synth_feat if self.uses_feat else None, train_synth_dmap if self.uses_dmap else None,
                        train_pos_ts if self.uses_ts else None, train_pos_feat if self.uses_feat else None, train_pos_dmap if self.uses_dmap else None), 
            batch_size=self.batch_size, shuffle=True
        )
        
        has_val = val_real_ts is not None and val_synth_ts is not None
        if has_val:
            val_loader = DataLoader(
                PairDataset(val_real_ts if self.uses_ts else None, val_real_feat if self.uses_feat else None, val_real_dmap if self.uses_dmap else None,
                            val_synth_ts if self.uses_ts else None, val_synth_feat if self.uses_feat else None, val_synth_dmap if self.uses_dmap else None,
                            val_pos_ts if self.uses_ts else None, val_pos_feat if self.uses_feat else None, val_pos_dmap if self.uses_dmap else None),
                batch_size=self.batch_size, shuffle=False
            )

        best_loss, best_state_emb, best_state_head, no_improve = float("inf"), None, None, 0
        
        def compute_late_fusion_diff(emb_r, emb_s):
            diffs = []
            for k in ['ts', 'feat', 'dmap']:
                if k in emb_r:
                    diffs.append(torch.abs(emb_r[k] - emb_s[k]))
            return torch.cat(diffs, dim=-1)

        for epoch in range(self.epochs):
            self.embedding.train()
            self.head.train()
            for r, s, label in train_loader:
                emb_r = self.embedding(r[0], r[1], r[2])
                emb_s = self.embedding(s[0], s[1], s[2])
                
                diff_concat = compute_late_fusion_diff(emb_r, emb_s)
                logits = self.head(diff_concat)
                
                loss = criterion(logits, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if has_val:
                self.embedding.eval()
                self.head.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for r, s, label in val_loader:
                        emb_r = self.embedding(r[0], r[1], r[2])
                        emb_s = self.embedding(s[0], s[1], s[2])
                        
                        diff_concat = compute_late_fusion_diff(emb_r, emb_s)
                        logits = self.head(diff_concat)
                        loss = criterion(logits, label.float())
                        val_loss += loss.item() * len(label)
                val_loss /= len(val_loader.dataset)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state_emb = copy.deepcopy(self.embedding.state_dict())
                    best_state_head = copy.deepcopy(self.head.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience: break

        if best_state_emb is not None: 
            self.embedding.load_state_dict(best_state_emb)
            self.head.load_state_dict(best_state_head)

    def evaluate(self, audit_real_ts, audit_real_feat, audit_real_dmap,
                 audit_synth_ts, audit_synth_feat, audit_synth_dmap,
                 leak_labels=None):
        """
        Rank each synthetic query against the full real gallery.

        If ``leak_labels`` is provided (1 = leaky / risk-relevant query), also
        report top-k accuracy and MRR averaged only over that subset. Gallery
        size is unchanged; only the query average is restricted.
        """
        self.embedding.eval()
        self.head.eval()
        with torch.no_grad():
            def embed(t, f, d, num_samples):
                t_t = torch.as_tensor(t, dtype=torch.float32) if self.uses_ts and t is not None and t.shape[-1] > 0 else torch.empty((num_samples, 0))
                f_t = torch.as_tensor(f, dtype=torch.float32) if self.uses_feat and f is not None and f.shape[-1] > 0 else torch.empty((num_samples, 0))
                d_t = torch.as_tensor(d, dtype=torch.float32) if self.uses_dmap and d is not None and d.shape[-1] > 0 else torch.empty((num_samples, 0))
                
                res = self.embedding(t_t, f_t, d_t)
                return {k: v.cpu() for k, v in res.items()}

            real_embs = embed(audit_real_ts, audit_real_feat, audit_real_dmap, len(audit_real_ts) if audit_real_ts is not None else len(audit_real_feat))
            synth_embs = embed(audit_synth_ts, audit_synth_feat, audit_synth_dmap, len(audit_synth_ts) if audit_synth_ts is not None else len(audit_synth_feat))

            n_queries = len(list(synth_embs.values())[0])
            n_targets = len(list(real_embs.values())[0])
            similarity_matrix = np.zeros((n_queries, n_targets))

            eval_batch_size = 64
            for i in range(0, n_queries, eval_batch_size):
                end = min(i + eval_batch_size, n_queries)
                
                diffs = []
                # Ensure isolated modality-wise differences before concatenation
                for k in ['ts', 'feat', 'dmap']:
                    if k in real_embs:
                        s_batch = synth_embs[k][i:end].unsqueeze(1).expand(-1, n_targets, -1)
                        r_batch = real_embs[k].unsqueeze(0).expand(end - i, -1, -1)
                        diffs.append(torch.abs(s_batch - r_batch))
                        
                diff_concat = torch.cat(diffs, dim=-1)
                logits = self.head(diff_concat)
                similarity_matrix[i:end] = logits.numpy()

        ranks = []
        for i in range(n_queries):
            true_score = similarity_matrix[i, i]
            rank = np.sum(similarity_matrix[i, :] > true_score) + 1
            ranks.append(rank)

        ranks = np.asarray(ranks, dtype=np.float64)
        metrics = {}
        for k in self.top_k:
            acc = float(np.mean(ranks <= k))
            base_rate = k / n_targets
            metrics[f"top{k}_accuracy"] = acc
            metrics[f"top{k}_lift"] = acc / base_rate
        metrics["mrr"] = float(np.mean(1.0 / ranks))
        metrics["n_targets"] = n_queries

        # Leaky-subset metrics (same gallery; average only over risk-relevant queries).
        for k in self.top_k:
            metrics[f"top{k}_accuracy_leaky"] = float("nan")
        metrics["mrr_leaky"] = float("nan")
        metrics["n_leaky"] = 0

        if leak_labels is not None:
            leak_mask = np.asarray(leak_labels).astype(bool)
            if leak_mask.shape[0] != n_queries:
                raise ValueError(
                    f"leak_labels length {leak_mask.shape[0]} != n_queries {n_queries}."
                )
            leak_ranks = ranks[leak_mask]
            n_leaky = int(leak_ranks.shape[0])
            metrics["n_leaky"] = n_leaky
            if n_leaky > 0:
                for k in self.top_k:
                    metrics[f"top{k}_accuracy_leaky"] = float(np.mean(leak_ranks <= k))
                metrics["mrr_leaky"] = float(np.mean(1.0 / leak_ranks))
        return metrics

class BlackboxReID(SiameseReID): name = "blackbox"; uses_ts = True
class GreyboxIReID(SiameseReID): name = "greybox1"; uses_ts = True; uses_feat = True
class GreyboxIIReID(SiameseReID): name = "greybox2"; uses_ts = True; uses_feat = True; uses_dmap = True
attacks = {"blackbox": BlackboxReID, "greybox1": GreyboxIReID, "greybox2": GreyboxIIReID}
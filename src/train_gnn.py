"""Heterogeneous GraphSAGE for supplier risk.

Nodes  supplier (features), bank account, address, director, buyer
Edges  supplier-bank, supplier-address, supplier-director, supplier-buyer,
       supplier-supplier co-bidding (+ reverse edges)

  python train_gnn.py               GraphSAGE, averaged over model seeds
  python train_gnn.py --no-graph    same network without edges (ablation)
  python train_gnn.py --stack       out-of-fold GNN scores for the hybrid XGBoost
"""
import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from common import (GOLD, RESULTS, SEED, evaluate, feature_matrix, load_features,
                    load_truth, save_metrics, split)


def _std(x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    mu, sd = x[rows].mean(0), x[rows].std(0)
    return (x - mu) / np.where(sd > 0, sd, 1.0)


def build_graph(df: pd.DataFrame, train_rows: np.ndarray) -> HeteroData:
    data = HeteroData()
    X = feature_matrix(df, links=True).values.astype(np.float32)
    data["supplier"].x = torch.tensor(_std(X, train_rows), dtype=torch.float)
    data["supplier"].y = torch.tensor(df["confirmed_fraud"].values, dtype=torch.float)
    sid = pd.Series(np.arange(len(df)), index=df["supplier_id"])

    def attach(table, key, node_type, relation):
        e = pd.read_parquet(GOLD / table)[["supplier_id", key]].dropna()
        e = e[e["supplier_id"].isin(sid.index)]
        codes, uniques = pd.factorize(e[key])
        deg = np.bincount(codes).astype(np.float32)
        data[node_type].x = torch.tensor(np.log1p(deg)[:, None], dtype=torch.float)
        data["supplier", relation, node_type].edge_index = torch.tensor(
            np.vstack([sid[e["supplier_id"]].values, codes]), dtype=torch.long)

    attach("edges_supplier_bank", "bank_account", "bank", "pays_into")
    attach("edges_supplier_address", "address_key", "address", "registered_at")
    attach("edges_supplier_director", "director_din", "director", "directed_by")
    attach("edges_supplier_buyer", "buyer_id", "buyer", "invoices")

    cb = pd.read_parquet(GOLD / "edges_co_bid")
    a, b = sid[cb["supplier_a"]].values, sid[cb["supplier_b"]].values
    data["supplier", "co_bids", "supplier"].edge_index = torch.tensor(
        np.vstack([np.r_[a, b], np.r_[b, a]]), dtype=torch.long)
    return T.ToUndirected()(data)


class SupplierGNN(nn.Module):
    def __init__(self, data: HeteroData, hidden=64, layers=2, dropout=0.3, use_graph=True):
        super().__init__()
        node_types, edge_types = data.metadata()
        self.use_graph, self.dropout = use_graph, dropout
        self.proj = nn.ModuleDict({nt: nn.Linear(data[nt].num_features, hidden)
                                   for nt in node_types})
        self.convs = nn.ModuleList([
            HeteroConv({et: SAGEConv(hidden, hidden) for et in edge_types}, aggr="sum")
            for _ in range(layers)])
        self.norms = nn.ModuleList([nn.ModuleDict({nt: nn.LayerNorm(hidden)
                                                   for nt in node_types})
                                    for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x_dict, edge_index_dict):
        h = {nt: F.relu(self.proj[nt](x)) for nt, x in x_dict.items()}
        h_self = h["supplier"]
        if self.use_graph:
            for conv, norm in zip(self.convs, self.norms):
                out = conv(h, edge_index_dict)
                h = {nt: h[nt] + F.dropout(F.relu(norm[nt](out[nt])), self.dropout,
                                           self.training) if nt in out else h[nt]
                     for nt in h}
        return self.head(torch.cat([h_self, h["supplier"]], dim=-1)).squeeze(-1)


def fit_predict(data, train_mask, val_mask, seed, use_graph, epochs=500, patience=60):
    """Train on train_mask with early stopping on val PR-AUC; return scores for all suppliers."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SupplierGNN(data, use_graph=use_graph).to(data["supplier"].x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    y = data["supplier"].y
    pos_weight = (train_mask.sum() - y[train_mask].sum()) / y[train_mask].sum().clamp(min=1)

    best, best_state, wait = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask],
                                                  pos_weight=pos_weight)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(data.x_dict, data.edge_index_dict))
        ap = average_precision_score(y[val_mask].cpu(), p[val_mask].cpu())
        if ap > best:
            best, best_state, wait = ap, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(data.x_dict, data.edge_index_dict)).cpu().numpy()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_features()
    s = split(df, args.split_seed)
    tr, va, te = [(s == k).values for k in ["train", "val", "test"]]
    data = build_graph(df, tr).to(device)
    print(f"device: {device}\n{data}")
    trm, vam = torch.tensor(tr, device=device), torch.tensor(va, device=device)
    test = load_truth(df)[te].assign(confirmed_fraud=df["confirmed_fraud"].values[te])

    if args.stack:
        # Out-of-fold scores on train, fold-averaged scores on val/test -> no leakage
        y = df["confirmed_fraud"].values
        train_idx = np.where(tr)[0]
        score = np.zeros(len(df))
        folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for f, (fit_i, hold_i) in enumerate(folds.split(train_idx, y[train_idx])):
            fit = np.zeros(len(df), dtype=bool)
            fit[train_idx[fit_i]] = True
            p = fit_predict(data, torch.tensor(fit, device=device), vam, SEED + f, True)
            score[train_idx[hold_i]] = p[train_idx[hold_i]]
            score[~tr] += p[~tr] / 5
            print(f"  fold {f + 1}/5 done")
        pd.DataFrame({"supplier_id": df["supplier_id"], "gnn_score": score}).to_parquet(
            RESULTS / f"gnn_stack_scores_split{args.split_seed}.parquet", index=False)
        print(f"wrote gnn_stack_scores_split{args.split_seed}.parquet")
        return

    name = "mlp_no_graph" if args.no_graph else "graphsage"
    runs = []
    for seed in range(args.seeds):
        p = fit_predict(data, trm, vam, seed, not args.no_graph)
        runs.append((evaluate(test, p[te]), p[te]))
        print(f"  seed {seed}: PR-AUC {runs[-1][0]['pr_auc']:.3f}")
    metrics = {k: float(np.nanmean([m[k] for m, _ in runs])) for k in runs[0][0]}
    metrics["pr_auc_std"] = float(np.std([m["pr_auc"] for m, _ in runs]))
    metrics["seeds"] = args.seeds
    save_metrics(name, metrics, args.split_seed)
    if args.split_seed != 0:
        return
    pd.DataFrame({"supplier_id": df.loc[te, "supplier_id"].values,
                  f"{name}_score": np.mean([p for _, p in runs], axis=0)}
                 ).to_parquet(RESULTS / f"scores_{name}.parquet", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--stack", action="store_true")
    ap.add_argument("--seeds", type=int, default=3, help="model seeds per split")
    ap.add_argument("--split-seed", type=int, default=0)
    main(ap.parse_args())

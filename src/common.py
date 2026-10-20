"""Shared data loading, supplier-level split and evaluation for every model.

Training and validation use AUDIT labels only (what a fraud team really has).
Final test evaluation uses the simulator's GROUND TRUTH, so models are also graded
on fraud the audits missed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
TRUTH = ROOT / "data" / "ground_truth" / "supplier_truth.csv"
RESULTS = ROOT / "results"
SEED = 42
REVIEW_BUDGET = 0.10          # the team can investigate the top 10% of suppliers
SCHEMES = ["shell_ring", "cartel", "split", "inflate"]

TABULAR = ["supplier_age_days", "is_service", "n_invoices", "total_amount", "n_buyers",
           "top_buyer_share", "near_limit_share", "burst_share", "median_price_ratio",
           "p90_price_ratio", "unshipped_share", "n_bids", "win_rate",
           "mean_bid_to_estimate", "mean_lost_bid_gap", "close_loss_share", "n_directors"]
# Relational features: links to other suppliers via shared attributes or repeated co-bidding
LINK_COUNTS = ["shared_bank_count", "shared_address_count", "shared_building_count",
               "shared_director_count", "n_bid_partners", "top_partner_share"]
LOG = ["n_invoices", "total_amount", "n_bids", "supplier_age_days"]


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(GOLD / "supplier_features").sort_values("supplier_id")
    return df.reset_index(drop=True)


def feature_matrix(df: pd.DataFrame, links: bool = True) -> pd.DataFrame:
    X = df[TABULAR + (LINK_COUNTS if links else [])].astype(float).copy()
    for c in LOG:
        X[c] = np.log1p(X[c].clip(lower=0))
    cats = pd.get_dummies(df["category"], prefix="category", dtype=float)
    return pd.concat([X, cats], axis=1)


def split(df: pd.DataFrame, seed: int = 0) -> pd.Series:
    """60/20/20 supplier-level split, stratified on audit labels. Members of one
    fraud ring can land in different splits, as in reality, where investigators
    know part of a ring."""
    idx = np.arange(len(df))
    y = df["confirmed_fraud"].values
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    s = pd.Series("train", index=idx)
    s[va], s[te] = "val", "test"
    return s


def load_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Ground truth aligned to df. Use ONLY for final test evaluation."""
    t = pd.read_csv(TRUTH)
    return df[["supplier_id"]].merge(t, on="supplier_id", how="left")


def evaluate(test_df: pd.DataFrame, scores: np.ndarray) -> dict:
    """test_df needs is_fraud, scheme (ground truth) and confirmed_fraud (audit)."""
    y = test_df["is_fraud"].values
    k = int(round(REVIEW_BUDGET * len(scores)))
    top = np.zeros(len(scores), dtype=bool)
    top[np.argsort(-scores)[:k]] = True
    out = {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc_vs_audit_labels": float(average_precision_score(test_df["confirmed_fraud"], scores)),
        "precision_at_budget": float(y[top].mean()),
        "recall_at_budget": float(top[y == 1].mean()),
    }
    for s in SCHEMES:
        m = (test_df["scheme"] == s).values
        out[f"recall_{s}"] = float(top[m].mean()) if m.any() else float("nan")
    unaudited = (y == 1) & (test_df["confirmed_fraud"].values == 0)
    out["recall_on_fraud_audits_missed"] = float(top[unaudited].mean())
    return out


def save_metrics(name: str, metrics: dict, split_seed: int):
    runs = RESULTS / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{name}__split{split_seed}.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[{name}, split {split_seed}]")
    for k, v in metrics.items():
        print(f"  {k:32s} {v:.3f}" if isinstance(v, float) else f"  {k:32s} {v}")

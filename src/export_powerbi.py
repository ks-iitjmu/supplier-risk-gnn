"""Export clean CSVs for the Power BI dashboard into powerbi/data/.

  supplier_risk.csv      one row per held-out test supplier (split 0): risk scores,
                         rank, top SHAP reasons, key features, audit label and the
                         simulator's ground truth (for the demo only)
  model_performance.csv  overall metrics per model, mean and std over 5 splits
  scheme_recall.csv      recall per fraud scheme per model, mean and std
  shap_importance.csv    global SHAP importance per XGBoost model

All four tables are also written as sheets of supplier_risk_dashboard.xlsx, which the
browser version of Power BI loads as one dataset.

Run after run_models.sh:  python src/export_powerbi.py
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from common import RESULTS, ROOT, load_features, load_truth, split

OUT = ROOT / "powerbi" / "data"
MODEL_LABELS = {"xgb_tabular": "XGBoost (tabular)", "xgb_links": "XGBoost (+ relational)",
                "mlp_no_graph": "Neural net (no graph)", "graphsage": "GraphSAGE",
                "xgb_hybrid": "Hybrid (XGBoost + GraphSAGE)"}
METRIC_LABELS = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC",
                 "precision_at_budget": "Precision at top 10%",
                 "recall_at_budget": "Recall at top 10%",
                 "recall_on_fraud_audits_missed": "Recall on fraud audits missed"}
SCHEME_LABELS = {"recall_shell_ring": "Shell rings", "recall_cartel": "Cartels",
                 "recall_split": "Split invoicing", "recall_inflate": "Price inflation"}
SCHEME_NAMES = {"none": "Not fraud", "shell_ring": "Shell ring", "cartel": "Cartel",
                "split": "Split invoicing", "inflate": "Price inflation"}


def supplier_risk() -> pd.DataFrame:
    df = load_features()
    te = (split(df, 0) == "test").values
    t = df[te].merge(load_truth(df)[["supplier_id", "is_fraud", "scheme"]], on="supplier_id")
    for f in ["scores_xgb_links", "scores_xgb_hybrid", "scores_graphsage", "reasons_xgb_links"]:
        if (RESULTS / f"{f}.parquet").exists():
            t = t.merge(pd.read_parquet(RESULTS / f"{f}.parquet"), on="supplier_id", how="left")
    t = t.rename(columns={"xgb_links_score": "risk_score", "xgb_hybrid_score": "hybrid_score",
                          "graphsage_score": "graphsage_score", "top_reasons": "top_reasons"})
    t["risk_rank"] = t["risk_score"].rank(ascending=False, method="first").astype(int)
    t["in_review_queue"] = t["risk_rank"] <= int(round(0.10 * len(t)))
    t["audit_status"] = np.where(t["confirmed_fraud"] == 1, "Confirmed fraud", "Not confirmed")
    t["true_scheme"] = t["scheme"].map(SCHEME_NAMES)
    keep = ["supplier_id", "category", "risk_rank", "risk_score", "hybrid_score",
            "graphsage_score", "in_review_queue", "top_reasons", "audit_status", "is_fraud",
            "true_scheme", "n_invoices", "total_amount", "n_buyers", "near_limit_share",
            "p90_price_ratio", "unshipped_share", "n_bids", "top_partner_share",
            "close_loss_share", "shared_bank_count", "shared_address_count",
            "shared_director_count", "supplier_age_days"]
    t = t[[c for c in keep if c in t.columns]].sort_values("risk_rank")
    t["total_amount"] = t["total_amount"].round(0)
    return t.round(4)


def run_metrics():
    runs = defaultdict(list)
    for f in sorted((RESULTS / "runs").glob("*.json")):
        runs[f.stem.split("__")[0]].append(json.loads(f.read_text()))
    perf, scheme = [], []
    for model, rs in runs.items():
        for key, label in {**METRIC_LABELS, **SCHEME_LABELS}.items():
            v = np.array([r[key] for r in rs], dtype=float)
            row = {"model": MODEL_LABELS.get(model, model), "mean": round(np.nanmean(v), 4),
                   "std": round(np.nanstd(v), 4), "n_splits": len(rs)}
            if key in METRIC_LABELS:
                perf.append({**row, "metric": label})
            else:
                scheme.append({**row, "scheme": label})
    return pd.DataFrame(perf), pd.DataFrame(scheme)


def shap_importance() -> pd.DataFrame:
    frames = []
    for model in ["xgb_tabular", "xgb_links", "xgb_hybrid"]:
        f = RESULTS / f"shap_{model}.csv"
        if f.exists():
            frames.append(pd.read_csv(f).assign(model=MODEL_LABELS[model]))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    perf, scheme = run_metrics()
    tables = {"supplier_risk": supplier_risk(), "model_performance": perf,
              "scheme_recall": scheme, "shap_importance": shap_importance()}
    with pd.ExcelWriter(OUT / "supplier_risk_dashboard.xlsx") as xl:
        for name, df in tables.items():
            df.to_csv(OUT / f"{name}.csv", index=False)
            df.to_excel(xl, sheet_name=name, index=False)
            print(f"{name:18s} {len(df):>5} rows")
    print("written to powerbi/data/ (4 CSVs + supplier_risk_dashboard.xlsx)")

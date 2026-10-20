"""XGBoost variants, explained with SHAP.

  --variant tabular   invoice, pricing, shipment and bidding features only
  --variant links     + counts of suppliers sharing bank account, address, director
  --variant hybrid    + out-of-fold GraphSAGE score (run train_gnn.py --stack first)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from common import (RESULTS, SEED, evaluate, feature_matrix, load_features, load_truth,
                    save_metrics, split)


def main(variant: str, gpu: bool, split_seed: int):
    df = load_features()
    X = feature_matrix(df, links=variant != "tabular")
    if variant == "hybrid":
        g = pd.read_parquet(RESULTS / f"gnn_stack_scores_split{split_seed}.parquet")
        X["gnn_score"] = df[["supplier_id"]].merge(g, on="supplier_id")["gnn_score"].values
    y = df["confirmed_fraud"].values
    s = split(df, split_seed)
    tr, va, te = [(s == k).values for k in ["train", "val", "test"]]

    pos = y[tr].sum()
    model = XGBClassifier(
        n_estimators=1000, max_depth=4, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=2, scale_pos_weight=(tr.sum() - pos) / pos,
        eval_metric="aucpr", early_stopping_rounds=100, tree_method="hist",
        device="cuda" if gpu else "cpu", random_state=SEED)
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)

    p = model.predict_proba(X[te])[:, 1]
    test = load_truth(df)[te].assign(confirmed_fraud=y[te])
    name = f"xgb_{variant}"
    save_metrics(name, evaluate(test, p), split_seed)
    if split_seed != 0:  # scores, SHAP and reasons are kept for split 0 (dashboard)
        return

    pd.DataFrame({"supplier_id": df.loc[te, "supplier_id"].values, f"{name}_score": p}
                 ).to_parquet(RESULTS / f"scores_{name}.parquet", index=False)

    sv = shap.TreeExplainer(model).shap_values(X[te])
    names = np.array([c.replace("_", " ") for c in X.columns])
    imp = (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)})
           .sort_values("mean_abs_shap", ascending=False))
    imp.to_csv(RESULTS / f"shap_{name}.csv", index=False)
    top = imp.head(15)[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#2b6cb0")
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Drivers of supplier risk ({name})")
    plt.tight_layout()
    plt.savefig(RESULTS / f"shap_{name}.png", dpi=150)
    plt.close()

    reasons = []
    for row in sv:
        idx = np.argsort(-row)[:3]
        reasons.append(", ".join(names[idx[row[idx] > 0]]))
    pd.DataFrame({"supplier_id": df.loc[te, "supplier_id"].values, "top_reasons": reasons}
                 ).to_parquet(RESULTS / f"reasons_{name}.parquet", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="links", choices=["tabular", "links", "hybrid"])
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args()
    main(a.variant, a.gpu, a.split_seed)

"""Measure address resolution against the simulator's hidden ground truth.
Evaluation only: the pipeline itself never reads ground truth."""
from itertools import combinations
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / "data"
silver = pd.read_parquet(D / "silver/suppliers")[["supplier_id", "address", "address_key",
                                                  "street_fixed"]]
truth = pd.read_csv(D / "ground_truth/supplier_truth.csv")[["supplier_id", "address_key"]]
m = silver.merge(truth, on="supplier_id", suffixes=("_resolved", "_true"))
m["address_key_true"] = m["address_key_true"].str.upper()
m["correct"] = m["address_key_resolved"] == m["address_key_true"]


def pairs(col):
    out = set()
    for _, g in m.groupby(col):
        out |= set(combinations(sorted(g["supplier_id"]), 2))
    return out


true_pairs, found = pairs("address_key_true"), pairs("address_key_resolved")
hit = len(true_pairs & found)
print(f"Typos corrected by fuzzy matching: {int(m['street_fixed'].sum())}")
print(f"Address keys resolved correctly:   {m['correct'].mean():.2%} of {len(m):,} suppliers")
print(f"Shared-address links  found {len(found)} | true {len(true_pairs)} | "
      f"precision {hit / max(len(found), 1):.1%} | recall {hit / max(len(true_pairs), 1):.1%}")
bad = m[~m["correct"]]
if len(bad):
    print("\nExamples of unresolved addresses:")
    print(bad[["address", "address_key_resolved", "address_key_true"]].head(5).to_string())

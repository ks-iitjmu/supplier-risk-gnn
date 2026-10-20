"""Check that each scheme leaves a signal AND that legit confounders overlap with it.
If a single feature separates fraud perfectly, the simulation is too easy."""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / "data"
sup = pd.read_csv(D / "raw/suppliers.csv")
dirs = pd.read_csv(D / "raw/supplier_directors.csv")
buy = pd.read_csv(D / "raw/buyers.csv")
inv = pd.read_csv(D / "raw/invoices.csv")
shp = pd.read_csv(D / "raw/shipments.csv")
prod = pd.read_csv(D / "raw/products.csv")
bids = pd.read_csv(D / "raw/bids.csv")
lab = pd.read_csv(D / "raw/audit_labels.csv")
gt = pd.read_csv(D / "ground_truth/supplier_truth.csv")

print(f"suppliers {len(sup):,} | invoices {len(inv):,} | shipments {len(shp):,} | "
      f"tenders {bids.tender_id.nunique():,} | bids {len(bids):,}")
print("\nSuppliers per scheme:\n", gt.scheme.value_counts().to_string())
t = gt.merge(lab, on="supplier_id")
print(f"\nTrue fraud rate {t.is_fraud.mean():.2%} | audit-confirmed {t.confirmed_fraud.mean():.2%} | "
      f"fraud caught by audits {t.loc[t.is_fraud == 1, 'confirmed_fraud'].mean():.0%}")

# Per-supplier tabular signals
inv = inv.merge(buy[["buyer_id", "approval_limit"]], on="buyer_id").merge(
    prod[["product_id", "market_unit_price"]], on="product_id")
inv["near_limit"] = inv.amount.between(0.85 * inv.approval_limit, inv.approval_limit, inclusive="left")
inv["price_ratio"] = inv.unit_price / inv.market_unit_price
recv = shp.groupby("invoice_id").quantity_received.sum()
inv["recv_ratio"] = inv.invoice_id.map(recv).fillna(0) / inv.quantity
f = inv.groupby("supplier_id").agg(
    near_limit_share=("near_limit", "mean"),
    median_price_ratio=("price_ratio", "median"),
    p90_price_ratio=("price_ratio", lambda x: x.quantile(0.9)),
    unshipped_share=("recv_ratio", lambda x: (x < 0.5).mean()),
    n_invoices=("invoice_id", "count"))
f = gt.merge(f, on="supplier_id", how="left")
print("\nTabular signals (median per group):")
print(f.groupby("scheme")[["near_limit_share", "p90_price_ratio", "unshipped_share",
                           "n_invoices"]].median().round(3).to_string())

# Relational signals: exact sharing of bank account / director
acct_n = sup.groupby("bank_account").supplier_id.transform("count")
din_n = dirs.groupby("director_din").supplier_id.transform("count")
shared_dir = dirs.loc[din_n > 1, "supplier_id"].unique()
r = gt.assign(shares_bank=(acct_n > 1).values, shares_director=gt.supplier_id.isin(shared_dir))
print("\nShare of suppliers with a shared bank account / director:")
print(r.groupby("scheme")[["shares_bank", "shares_director"]].mean().round(3).to_string())
legit_linked = r[(r.scheme == "none") & (r.shares_bank | r.shares_director)]
print(f"legit suppliers with shared links (confounder): {len(legit_linked)}")

# Address: exact raw strings rarely match - entity resolution is needed
same_raw = sup.address.duplicated(keep=False).mean()
same_true = gt.address_key.duplicated(keep=False).mean()
print(f"\nAddresses: {same_raw:.1%} share an exact raw string, "
      f"{same_true:.1%} truly share a location (needs entity resolution)")

# Tenders: co-bidding and win concentration
w = bids.merge(gt[["supplier_id", "scheme"]], on="supplier_id")
print("\nBidding (median per group):")
print(w.groupby(["scheme", "supplier_id"]).agg(n_bids=("tender_id", "count"),
                                               win_rate=("is_winner", "mean"))
      .groupby("scheme").median().round(3).to_string())

# Fronts: ring members that look clean on invoices
ring = f[f.scheme == "shell_ring"]
print(f"\nShell-ring members with no unshipped invoices (clean-looking fronts): "
      f"{(ring.unshipped_share.fillna(0) < 0.05).mean():.0%}")

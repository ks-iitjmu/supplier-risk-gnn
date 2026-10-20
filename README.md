# Supplier Risk Detection with Graph Neural Networks

Detects procurement fraud (shell-company rings, bid-rigging cartels, split invoicing,
price inflation) in a simulated procurement network of suppliers, buyers, invoices,
shipments and tenders.

## Why simulated data

Real procurement fraud data is confidential, so no public dataset contains suppliers,
invoices and shipments with fraud labels. The simulator encodes four schemes from the
ACFE fraud tree, plus deliberate confounders so no single feature separates fraud.
The same approach is used in anti-money-laundering research (e.g. IBM AMLSim).

## Status

- [x] Step 1: simulator (`simulator/simulate.py`) and sanity check
- [ ] Step 2: PySpark bronze/silver/gold pipeline (local + Databricks)
- [ ] Step 3: supplier-level split
- [ ] Step 4: XGBoost, no-graph ablation, GraphSAGE, hybrid
- [ ] Step 5: evaluation with recall per fraud scheme, SHAP
- [ ] Step 6: Power BI dashboard
- [ ] Step 7: write-up

## Step 1: generate the data

```bash
python simulator/simulate.py          # ~1 minute, deterministic with --seed
python simulator/sanity_check.py      # verifies signals and confounders
```

`data/raw/` holds what a fraud team would have. `data/ground_truth/` holds the hidden
answers and is used **only** for final evaluation, never for training.

| Scheme | How it works | Where the signal is | Confounder that hides it |
|---|---|---|---|
| Shell ring | 3-6 new suppliers share bank account, address or director; bill for goods that never ship; ~1/4 are clean-looking fronts | links between suppliers, missing goods receipts | legit corporate groups share directors/accounts; service suppliers have no goods receipts |
| Cartel | established suppliers rig tenders at one buyer: all bid, winners rotate, cover bids just above winner | co-bidding pattern, inflated prices at one buyer | legit frequent bidders, premium-priced legit suppliers |
| Split invoicing | orders split into invoices just under the approval limit | near-limit invoice share, same-day bursts | legit contracts priced at the limit |
| Price inflation | 25-60% above market on part of the invoices | price vs market | premium-brand legit suppliers |

Audit labels catch only about half of the fraud, as in reality. Models train on
audit labels and are graded on ground truth.

Sample run (seed 7): 5,000 suppliers, ~188K invoices, ~2.5K tenders, 6.3% true
fraud, ~3% audit-confirmed.

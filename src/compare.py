"""Aggregate results/runs/*.json over split seeds into mean ± std tables."""
import json
from collections import defaultdict

import numpy as np

from common import RESULTS

ORDER = ["xgb_tabular", "xgb_links", "mlp_no_graph", "graphsage", "xgb_hybrid"]
OVERALL = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"),
           ("precision_at_budget", "Precision @ top 10%"),
           ("recall_at_budget", "Recall @ top 10%"),
           ("recall_on_fraud_audits_missed", "Recall on fraud audits missed")]
SCHEMES = [("recall_shell_ring", "Shell rings"), ("recall_cartel", "Cartels"),
           ("recall_split", "Split invoicing"), ("recall_inflate", "Price inflation")]

runs = defaultdict(list)
for f in sorted((RESULTS / "runs").glob("*.json")):
    runs[f.stem.split("__")[0]].append(json.loads(f.read_text()))
names = [n for n in ORDER if n in runs] + sorted(set(runs) - set(ORDER))
n_splits = max(len(v) for v in runs.values())


def cell(name, key):
    v = np.array([r[key] for r in runs[name]], dtype=float)
    return f"{np.nanmean(v):.3f} ± {np.nanstd(v):.3f}"


def table(cols, title):
    lines = [f"**{title}**", "", "| Model | " + " | ".join(h for _, h in cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    lines += [f"| {n} | " + " | ".join(cell(n, k) for k, _ in cols) + " |" for n in names]
    return "\n".join(lines)


out = (f"Mean ± standard deviation over {n_splits} train/val/test splits, "
       "graded on ground truth.\n\n" + table(OVERALL, "Overall") + "\n\n"
       + table(SCHEMES, "Recall per fraud scheme at a top-10% review budget"))
(RESULTS / "comparison.md").write_text(out + "\n")
print(out)

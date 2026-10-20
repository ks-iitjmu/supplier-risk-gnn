Mean ± standard deviation over 5 train/val/test splits, graded on ground truth.

**Overall**

| Model | PR-AUC | ROC-AUC | Precision @ top 10% | Recall @ top 10% | Recall on fraud audits missed |
|---|---|---|---|---|---|
| xgb_tabular | 0.454 ± 0.089 | 0.867 ± 0.029 | 0.348 ± 0.067 | 0.601 ± 0.113 | 0.569 ± 0.162 |
| xgb_links | 0.581 ± 0.059 | 0.909 ± 0.009 | 0.420 ± 0.025 | 0.725 ± 0.028 | 0.742 ± 0.086 |
| mlp_no_graph | 0.477 ± 0.040 | 0.869 ± 0.014 | 0.368 ± 0.020 | 0.635 ± 0.023 | 0.639 ± 0.072 |
| graphsage | 0.499 ± 0.062 | 0.874 ± 0.023 | 0.374 ± 0.039 | 0.644 ± 0.048 | 0.633 ± 0.091 |
| xgb_hybrid | 0.540 ± 0.098 | 0.915 ± 0.010 | 0.408 ± 0.044 | 0.703 ± 0.060 | 0.725 ± 0.080 |

**Recall per fraud scheme at a top-10% review budget**

| Model | Shell rings | Cartels | Split invoicing | Price inflation |
|---|---|---|---|---|
| xgb_tabular | 0.510 ± 0.117 | 0.644 ± 0.320 | 0.871 ± 0.072 | 0.506 ± 0.191 |
| xgb_links | 0.652 ± 0.050 | 0.809 ± 0.137 | 0.886 ± 0.074 | 0.611 ± 0.188 |
| mlp_no_graph | 0.627 ± 0.037 | 0.766 ± 0.141 | 0.582 ± 0.023 | 0.599 ± 0.064 |
| graphsage | 0.689 ± 0.029 | 0.744 ± 0.136 | 0.608 ± 0.068 | 0.549 ± 0.143 |
| xgb_hybrid | 0.726 ± 0.032 | 0.746 ± 0.208 | 0.771 ± 0.096 | 0.616 ± 0.115 |

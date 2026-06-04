# Experiment Run Log Template

Use this template to document every experimental run. Add completed entries to `docs/experiment_log/run_log.md` (to be created as runs progress).

## Run Metadata
* **Run ID**: `{dataset}_{pipeline}_{model}_{seed}_{version}` (e.g. `german_ctgan_xgboost_seed42_v1`)
* **Timestamp**: `YYYY-MM-DD HH:MM:SS`
* **Dataset**: `german_credit` / `gmsc_10k`
* **Seed**: `42` / `123` / `456` / `789` / `1337`
* **Pipeline**: `Real Baseline` / `CTGAN Synthetic` / `TVAE Synthetic`
* **System**: Intel i5-1335U, 8 GB RAM, CPU-only

## Configurations
* **Model**: XGBoost (Parameters: `max_depth`, `n_estimators`, `learning_rate`)
* **Generator Settings (if applicable)**: CTGAN/TVAE (Epochs: `300`, Batch Size: `500`)
* **Training Time**: `X` seconds

## Downstream Predictive Utility (Evaluated on Real Test Set)
* **ROC-AUC**: `0.XXXX`
* **F1-Score**: `0.XXXX`
* **Precision**: `0.XXXX`
* **Recall**: `0.XXXX`
* **Confusion Matrix**: `[[TN, FP], [FN, TP]]`

## SHAP Consistency (Compared with Real Baseline for current Seed)
* **Spearman Rank Correlation ($\rho$)**: `0.XXXX`
* **p-value**: `X.XXXe-XX`
* **Top 5 Features (Real)**: `[feat1, feat2, feat3, feat4, feat5]`
* **Top 5 Features (Synthetic)**: `[feat1, feat2, feat3, feat4, feat5]`

## Privacy Evaluation
* **Distance to Closest Record (DCR) - 5th Percentile**: `X.XX`
* **Nearest Neighbor Distance Ratio (NNDR) - 5th Percentile**: `X.XX`
* **Membership Inference Attack (MIA) AUC**: `0.XXXX`

## Notes & Observations
* *Add notes on stability, training anomalies, marginal distribution drift, or feature attribution shifts here.*

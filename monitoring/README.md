# AirPulse — Model Monitoring & Retraining

> **MLOps Practical Assessment — 10 Marks**  
> Extends the AirPulse PM2.5 forecasting project with model-quality monitoring and an automated retraining workflow.

---

## Overview

This directory implements **Tasks A–E** of the monitoring assessment. It is fully self-contained: no XKDR API token is needed because the new data batch is generated synthetically using the same 18-feature schema as the production model-ready data.

```
monitoring/
├── monitoring_retraining.ipynb   ← Jupyter notebook (all 5 tasks, with markdown cells)
├── run_monitoring_pipeline.py    ← Standalone Python script (identical logic)
├── quality_gate.py               ← Reusable quality-gate module
├── data/
│   └── new_batch.csv             ← Generated after first run
└── reports/
    ├── evaluation_report.json        ← Task B output
    ├── retrain_report.json           ← Task C output
    └── quality_gate_decisions.jsonl  ← Task D append-only decision log
```

---

## Quick Start

```bash
# From the REPO ROOT:
pip install -r requirements.txt

# Run the full pipeline (Tasks A → E)
python monitoring/run_monitoring_pipeline.py

# Or open the Jupyter notebook:
jupyter notebook monitoring/monitoring_retraining.ipynb
```

---

## Task Summary

### Task A — Prepare New Data
- Generates a realistic 1 200-row synthetic batch (3 stations × 400 hourly readings)
- Validates all 19 required columns (18 features + target)
- Drops rows with missing lag values and removes duplicate records
- Displays target (`target_pm25_next_hour`) distribution: min, median, mean, std, max, percentiles, per-station means

### Task B — Evaluate Existing Champion
- Loads `artifacts/models/residual_hgb_pipeline.joblib`
- Runs predictions on the new labelled batch
- Reports **MAE, RMSE, R²** vs. persistence baseline and per-station breakdown
- Saves `monitoring/reports/evaluation_report.json`

### Task C — Retrain Candidate Model
- Combines original training data (`data/processed/model_ready/X_train.csv`) + new batch  
  *(if original data is unavailable, trains on new batch alone — pipeline is identical)*
- Retrains using the **same** `PersistenceAnchorRegressor` + `HistGradientBoostingRegressor` pipeline
- Saves candidate to `artifacts/models/candidate_model.joblib`
- Logs full MLflow run to `AirPulse_PM25_Forecasting` experiment:
  - Params: all HGB hyperparams, row counts, feature count, run type
  - Metrics: `train_mae`, `train_rmse`, `train_r2`, `new_batch_mae`, `new_batch_rmse`, `new_batch_r2`
  - Artifacts: candidate model (sklearn flavour), new batch CSV

### Task D — Quality Gate

| Property | Value |
|----------|-------|
| **Primary metric** | MAE |
| **Decision rule** | `candidate_MAE ≤ champion_MAE × 1.02` |
| **If true** | Mark candidate ELIGIBLE FOR PROMOTION |
| **If false** | Retain champion |
| **Auto-overwrite** | **Never** — decision always recorded before any action |

#### Why MAE?
1. The HGB model is trained with `loss="absolute_error"` → MAE is the natural, internally-consistent gate metric
2. MAE is directly interpretable in µg/m³, mapping onto real-world health-advisory thresholds
3. RMSE is inflated by PM2.5 spikes common in Delhi; MAE degrades more predictably
4. R² can mislead when target distribution shifts seasonally

#### Why 2 % tolerance?
- Prevents over-promotion of marginally better models due to data-batch sampling noise
- Generous enough to capture genuine improvements across a 400-row new batch

#### Promotion flow (human-in-the-loop)
```
ELIGIBLE → human approves
         → mlflow.register_model(candidate_run_id)
         → client.set_registered_model_alias("airpulse-pm25-forecast", "champion", new_version)
         → redeploy FastAPI service
```

### Task E — Demonstration
- Prints a full execution summary table with all metrics and artifact paths
- Confirms all produced files exist with sizes

---

## Files Produced

| File | Description |
|------|-------------|
| `monitoring/data/new_batch.csv` | Validated new data batch |
| `monitoring/reports/evaluation_report.json` | Champion metrics on new batch |
| `monitoring/reports/retrain_report.json` | Candidate training details + MLflow run ID |
| `monitoring/reports/quality_gate_decisions.jsonl` | Append-only decision audit log |
| `artifacts/models/candidate_model.joblib` | Retrained candidate model artifact |

---

## Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Non-destructive** | Champion artifact is never overwritten; candidate saved separately |
| **Auditable** | Every gate decision appended to JSONL log (never modified) |
| **Internally consistent** | Gate metric (MAE) = model's training loss function |
| **Human-in-the-loop** | ELIGIBLE ≠ auto-deployed; human must approve MLflow alias update |
| **Reproducible** | Fixed random seed (2025), full MLflow run logged |
| **Modular** | `quality_gate.py` importable independently for use in CI/CD |

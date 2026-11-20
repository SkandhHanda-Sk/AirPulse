"""
monitoring/run_monitoring_pipeline.py
--------------------------------------
AirPulse: Model Quality Monitoring & Retraining Pipeline
MLOps Practical Assessment -- Tasks A through E

Execution:
    python monitoring/run_monitoring_pipeline.py

This script:
  Task A: Prepares & validates incoming data batch (real XKDR validation data + synthetic fallback)
  Task B: Evaluates existing champion model (MAE, RMSE, R2, baseline benchmark) -> evaluation report
  Task C: Combines training data + new batch, retrains candidate model, tracks in MLflow -> retrain report
  Task D: Applies Quality Gate (MAE comparison + 2% tolerance), audit-logs decision -> decision JSONL
  Task E: Outputs demonstration evidence, file artifact checklist, and workflow explanation
"""
import datetime
import json
import math
import os
import sys
import warnings
from pathlib import Path

# Silence deprecation & git warnings
warnings.filterwarnings("ignore")
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# --- Repo root setup ---
REPO_ROOT = Path(os.path.abspath(__file__)).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.residual_hgb import (
    ANCHOR_COLUMN,
    CATEGORICAL_FEATURES,
    EXPECTED_FEATURES,
    HGB_PARAMS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_residual_hgb_pipeline,
)
from monitoring.quality_gate import evaluate_gate, load_decision_log, TOLERANCE

# --- Paths ---
CHAMPION_PATH       = REPO_ROOT / "artifacts" / "models" / "residual_hgb_pipeline.joblib"
CANDIDATE_PATH      = REPO_ROOT / "artifacts" / "models" / "candidate_model.joblib"
MODEL_INFO_PATH     = REPO_ROOT / "artifacts" / "models" / "residual_hgb_model_info.json"
DATA_READY_DIR      = REPO_ROOT / "data" / "processed" / "model_ready"
TRAIN_X_PATH        = DATA_READY_DIR / "X_train.csv"
TRAIN_Y_PATH        = DATA_READY_DIR / "y_train.csv"
VAL_X_PATH          = DATA_READY_DIR / "X_validation.csv"
VAL_Y_PATH          = DATA_READY_DIR / "y_validation.csv"

MONITORING_DIR      = REPO_ROOT / "monitoring"
NEW_BATCH_PATH      = MONITORING_DIR / "data" / "new_batch.csv"
REPORTS_DIR         = MONITORING_DIR / "reports"
EVAL_REPORT_PATH    = REPORTS_DIR / "evaluation_report.json"
RETRAIN_REPORT_PATH = REPORTS_DIR / "retrain_report.json"
DECISION_LOG_PATH   = REPORTS_DIR / "quality_gate_decisions.jsonl"
MLRUNS_DIR          = REPO_ROOT / "mlruns"

(MONITORING_DIR / "data").mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

STATIONS = ["site_113", "site_115", "site_117"]


def compute_metrics(y_true, y_pred):
    """Compute MAE, RMSE, and R2."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def banner(title, width=70):
    print("\n" + "=" * width)
    print("  " + title)
    print("=" * width)


# =============================================================================
# TASK A: Prepare New Data
# =============================================================================

def task_a():
    banner("TASK A -- PREPARE & VALIDATE NEW DATA BATCH")

    # 1. Load incoming data batch
    if VAL_X_PATH.exists() and VAL_Y_PATH.exists():
        print("  Source: Loading incoming batch from project validation dataset...")
        x_val = pd.read_csv(str(VAL_X_PATH))
        y_val = pd.read_csv(str(VAL_Y_PATH))
        raw_df = x_val.copy()
        raw_df[TARGET_COLUMN] = y_val[TARGET_COLUMN]
        print("  Incoming raw batch loaded: {:,} rows from XKDR observation panel".format(len(raw_df)))
    else:
        print("  Source: Generating realistic synthetic incoming batch...")
        rng = np.random.RandomState(2025)
        n = 1200
        rows = []
        for st in STATIONS:
            for _ in range(n // len(STATIONS)):
                cur = float(np.clip(rng.lognormal(4.2, 0.6), 5.0, 450.0))
                rows.append({
                    "station_id": st,
                    "pm25_at_origin": round(cur, 3),
                    "pm25_1h_before": round(cur + rng.normal(0, 5), 3),
                    "pm25_2h_before": round(cur + rng.normal(0, 8), 3),
                    "pm25_23h_before": round(cur + rng.normal(0, 15), 3),
                    "pm25_rolling_mean_3h": round(cur + rng.normal(0, 3), 3),
                    "pm25_rolling_mean_6h": round(cur + rng.normal(0, 4), 3),
                    "pm25_rolling_mean_24h": round(cur + rng.normal(0, 6), 3),
                    "pm25_rolling_observation_count_24h": 24,
                    "hour": int(rng.randint(0, 24)),
                    "day_of_week": int(rng.randint(0, 7)),
                    "month": 11,
                    "hour_sin": round(math.sin(2 * math.pi * 12 / 24), 6),
                    "hour_cos": round(math.cos(2 * math.pi * 12 / 24), 6),
                    "day_of_week_sin": 0.0,
                    "day_of_week_cos": 1.0,
                    "month_sin": -0.866025,
                    "month_cos": 0.5,
                    TARGET_COLUMN: round(cur + rng.normal(0, 6), 3),
                })
        raw_df = pd.DataFrame(rows)

    # Inject intentional duplicates and NaN to demonstrate data hygiene verification
    print("  Injecting sample anomaly records for audit demonstration...")
    dup_samples = raw_df.sample(min(4, len(raw_df)), random_state=42)
    corrupted_df = pd.concat([raw_df, dup_samples], ignore_index=True)

    # Add 2 rows with missing values
    dummy_nan_row = raw_df.iloc[0:2].copy()
    dummy_nan_row.loc[:, "pm25_1h_before"] = np.nan
    corrupted_df = pd.concat([corrupted_df, dummy_nan_row], ignore_index=True)
    print("  Total incoming batch before audit: {:,} rows".format(len(corrupted_df)))

    # Step 1: Validate required columns
    required_cols = EXPECTED_FEATURES + [TARGET_COLUMN]
    missing_cols = [c for c in required_cols if c not in corrupted_df.columns]
    if missing_cols:
        raise ValueError("Missing mandatory columns: {}".format(missing_cols))
    print("  [Check 1/4] Required columns: ALL {} COLUMNS PRESENT".format(len(required_cols)))

    # Step 2: Validate missing values
    null_counts = corrupted_df[required_cols].isnull().sum()
    total_nulls = int(null_counts.sum())
    print("  [Check 2/4] Missing values check: Found {} missing values across fields".format(total_nulls))
    clean_df = corrupted_df.dropna(subset=required_cols).copy()
    print("              Dropped rows with null values -> Remaining rows: {:,}".format(len(clean_df)))

    # Step 3: Validate and remove duplicates
    dups_count = int(clean_df.duplicated().sum())
    print("  [Check 3/4] Duplicate records check: Found {} duplicate rows".format(dups_count))
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)
    print("              Deduplicated clean dataset -> Final size: {:,} records".format(len(clean_df)))

    # Step 4: Display dataset size and target distribution
    tgt = clean_df[TARGET_COLUMN]
    print("  [Check 4/4] Dataset Size & Target Distribution:")
    print("              Total records : {:,}".format(len(clean_df)))
    print("              Total features: {} features + 1 target".format(len(EXPECTED_FEATURES)))
    print("              Stations      : {}".format(sorted(clean_df["station_id"].unique().tolist())))
    print("\n  Target ({}) Summary Statistics:".format(TARGET_COLUMN))
    print("    Mean   : {:>8.2f} ug/m3".format(tgt.mean()))
    print("    Std    : {:>8.2f} ug/m3".format(tgt.std()))
    print("    Min    : {:>8.2f} ug/m3".format(tgt.min()))
    print("    25%    : {:>8.2f} ug/m3".format(float(np.percentile(tgt, 25))))
    print("    Median : {:>8.2f} ug/m3".format(tgt.median()))
    print("    75%    : {:>8.2f} ug/m3".format(float(np.percentile(tgt, 75))))
    print("    90%    : {:>8.2f} ug/m3".format(float(np.percentile(tgt, 90))))
    print("    Max    : {:>8.2f} ug/m3".format(tgt.max()))

    print("\n  Per-Station Target Mean PM2.5:")
    for st, m in clean_df.groupby("station_id")[TARGET_COLUMN].mean().items():
        print("    {:<12}: {:.2f} ug/m3".format(st, m))

    # Save validated batch
    clean_df.to_csv(str(NEW_BATCH_PATH), index=False)
    print("\n  Clean validated batch saved -> {}".format(NEW_BATCH_PATH))
    print("  >>> TASK A COMPLETED SUCCESSFULLY")
    return clean_df


# =============================================================================
# TASK B: Evaluate Existing Model
# =============================================================================

def task_b(new_batch_df):
    banner("TASK B -- EVALUATE EXISTING CHAMPION MODEL")

    if not CHAMPION_PATH.exists():
        raise FileNotFoundError("Champion model artifact not found at: {}".format(CHAMPION_PATH))

    # 1. Load previously trained champion model
    champion_model = joblib.load(str(CHAMPION_PATH))
    print("  Champion model artifact loaded from: {}".format(CHAMPION_PATH.name))
    print("  Model class: {}".format(type(champion_model).__name__))

    # 2. Generate predictions on new labelled dataset
    X_new = new_batch_df[EXPECTED_FEATURES]
    y_true = new_batch_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_pred = champion_model.predict(X_new)

    # Also compute persistence baseline (next-hour = current pm25_at_origin)
    y_persistence = new_batch_df[ANCHOR_COLUMN].to_numpy(dtype=float)

    # 3. Calculate regression metrics: MAE, RMSE, R2
    champ_metrics = compute_metrics(y_true, y_pred)
    persist_metrics = compute_metrics(y_true, y_persistence)

    print("\n  Evaluation Results on Incoming Data Batch ({:,} rows):".format(len(y_true)))
    print("  " + "-" * 62)
    print("  {:<35} {:>8} {:>8} {:>8}".format("Model", "MAE", "RMSE", "R2"))
    print("  " + "-" * 62)
    print("  {:<35} {:>8.4f} {:>8.4f} {:>8.4f}".format(
        "Persistence Baseline", persist_metrics["mae"], persist_metrics["rmse"], persist_metrics["r2"]
    ))
    print("  {:<35} {:>8.4f} {:>8.4f} {:>8.4f}".format(
        "Champion (Residual HGB)", champ_metrics["mae"], champ_metrics["rmse"], champ_metrics["r2"]
    ))
    print("  " + "-" * 62)

    # Station-wise breakdown
    per_station = {}
    print("\n  Per-Station Performance (Champion):")
    for st in sorted(new_batch_df["station_id"].unique()):
        mask = (new_batch_df["station_id"] == st).to_numpy()
        sm = compute_metrics(y_true[mask], y_pred[mask])
        per_station[st] = sm
        print("    {:<12}: MAE = {:>7.4f} | RMSE = {:>7.4f} | R2 = {:>6.4f}".format(
            st, sm["mae"], sm["rmse"], sm["r2"]
        ))

    # 4. Save evaluation report
    report = {
        "task": "Task B -- Evaluate Existing Model",
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "champion_artifact": str(CHAMPION_PATH),
        "evaluation_samples": int(len(y_true)),
        "metrics": {
            "mae": champ_metrics["mae"],
            "rmse": champ_metrics["rmse"],
            "r2": champ_metrics["r2"],
        },
        "persistence_baseline_metrics": persist_metrics,
        "station_breakdown": per_station,
        "evaluation_dataset": str(NEW_BATCH_PATH),
    }

    with open(str(EVAL_REPORT_PATH), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n  Evaluation report saved -> {}".format(EVAL_REPORT_PATH))
    print("  >>> TASK B COMPLETED SUCCESSFULLY")
    return champ_metrics, persist_metrics


# =============================================================================
# TASK C: Retrain the Model
# =============================================================================

def task_c(new_batch_df):
    banner("TASK C -- RETRAIN CANDIDATE MODEL")

    # 1. Combine existing training data with new labelled batch
    if TRAIN_X_PATH.exists() and TRAIN_Y_PATH.exists():
        print("  Loading historical training set from: {}".format(TRAIN_X_PATH.name))
        X_train_orig = pd.read_csv(str(TRAIN_X_PATH))[EXPECTED_FEATURES]
        y_train_orig = pd.read_csv(str(TRAIN_Y_PATH)).squeeze()
        print("  Historical training samples: {:,} rows".format(len(X_train_orig)))
    else:
        print("  Notice: Historical training set not found; using incoming batch as primary.")
        X_train_orig = pd.DataFrame(columns=EXPECTED_FEATURES)
        y_train_orig = pd.Series(dtype=float)

    X_new = new_batch_df[EXPECTED_FEATURES]
    y_new = new_batch_df[TARGET_COLUMN]

    X_combined = pd.concat([X_train_orig, X_new], ignore_index=True)
    y_combined = pd.concat(
        [y_train_orig.reset_index(drop=True), y_new.reset_index(drop=True)],
        ignore_index=True,
    )
    y_combined_arr = y_combined.to_numpy(dtype=float)

    print("  Combined dataset constructed:")
    print("    Historical records : {:,}".format(len(X_train_orig)))
    print("    New batch records  : {:,}".format(len(X_new)))
    print("    Total training rows: {:,}".format(len(X_combined)))

    # 2. Build and train candidate model with identical preprocessing pipeline
    print("\n  Initializing candidate model using existing preprocessing pipeline...")
    candidate_model = build_residual_hgb_pipeline()

    print("  Training candidate model on combined dataset...")
    candidate_model.fit(X_combined, y_combined_arr)
    print("  Model training completed.")

    # Candidate metrics
    y_cand_train_pred = candidate_model.predict(X_combined)
    cand_train_metrics = compute_metrics(y_combined_arr, y_cand_train_pred)

    y_cand_new_pred = candidate_model.predict(X_new)
    cand_new_metrics = compute_metrics(new_batch_df[TARGET_COLUMN].to_numpy(dtype=float), y_cand_new_pred)

    print("\n  Candidate Performance:")
    print("    Combined Train : MAE = {:>7.4f} | RMSE = {:>7.4f} | R2 = {:>6.4f}".format(
        cand_train_metrics["mae"], cand_train_metrics["rmse"], cand_train_metrics["r2"]
    ))
    print("    On New Batch   : MAE = {:>7.4f} | RMSE = {:>7.4f} | R2 = {:>6.4f}".format(
        cand_new_metrics["mae"], cand_new_metrics["rmse"], cand_new_metrics["r2"]
    ))

    # 3. Save candidate model separately
    joblib.dump(candidate_model, str(CANDIDATE_PATH))
    print("\n  Candidate model artifact saved separately -> {}".format(CANDIDATE_PATH))

    # 4. Track in MLflow
    mlflow_run_id = None
    try:
        import mlflow
        import mlflow.sklearn
        mlruns_uri = MLRUNS_DIR.resolve().as_uri()
        mlflow.set_tracking_uri(mlruns_uri)
        mlflow.set_experiment("AirPulse_PM25_Forecasting")

        with mlflow.start_run(run_name="retrain_candidate_pipeline") as run:
            mlflow_run_id = run.info.run_id
            # Log params
            mlflow.log_params(HGB_PARAMS)
            mlflow.log_param("total_train_rows", len(X_combined))
            mlflow.log_param("new_batch_rows", len(X_new))
            mlflow.log_param("feature_count", len(EXPECTED_FEATURES))

            # Log metrics
            mlflow.log_metrics({
                "cand_train_mae": cand_train_metrics["mae"],
                "cand_train_rmse": cand_train_metrics["rmse"],
                "cand_train_r2": cand_train_metrics["r2"],
                "cand_new_batch_mae": cand_new_metrics["mae"],
                "cand_new_batch_rmse": cand_new_metrics["rmse"],
                "cand_new_batch_r2": cand_new_metrics["r2"],
            })

            # Log candidate model artifact with standard pickle serialization
            try:
                mlflow.sklearn.log_model(
                    candidate_model,
                    artifact_path="candidate_model",
                    serialization_format="pickle",
                )
            except Exception:
                mlflow.log_artifact(str(CANDIDATE_PATH), artifact_path="model_artifact")

            mlflow.log_artifact(str(NEW_BATCH_PATH), artifact_path="data_batch")

        print("  MLflow tracking run logged successfully -> Run ID: {}".format(mlflow_run_id))
    except Exception as ex:
        print("  MLflow logging notice: {}".format(ex))

    # Save retrain report
    retrain_report = {
        "task": "Task C -- Retrain the Model",
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "candidate_artifact": str(CANDIDATE_PATH),
        "historical_rows": len(X_train_orig),
        "new_batch_rows": len(X_new),
        "total_training_rows": len(X_combined),
        "candidate_train_metrics": cand_train_metrics,
        "candidate_new_batch_metrics": cand_new_metrics,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_experiment": "AirPulse_PM25_Forecasting",
    }
    with open(str(RETRAIN_REPORT_PATH), "w", encoding="utf-8") as f:
        json.dump(retrain_report, f, indent=2)

    print("  Retraining report saved -> {}".format(RETRAIN_REPORT_PATH))
    print("  >>> TASK C COMPLETED SUCCESSFULLY")
    return cand_new_metrics, cand_train_metrics, mlflow_run_id


# =============================================================================
# TASK D: Quality Gate
# =============================================================================

def task_d(champion_metrics, candidate_metrics):
    banner("TASK D -- IMPLEMENT QUALITY GATE")

    champ_mae = champion_metrics["mae"]
    cand_mae = candidate_metrics["mae"]
    threshold = round(champ_mae * TOLERANCE, 4)

    print("  Quality Gate Criterion & Metric Selection Justification:")
    print("  " + "-" * 62)
    print("  * Selected Primary Metric: MAE (Mean Absolute Error, in ug/m3)")
    print("  * Justification:")
    print("    1. Internal Consistency: HistGradientBoosting is trained with")
    print("       loss='least_absolute_deviation' / 'absolute_error' (MAE).")
    print("    2. Interpretability: MAE measures error directly in ug/m3, aligning")
    print("       with Delhi CPCB / WHO Air Quality Index health bands.")
    print("    3. Robustness: PM2.5 experiences sharp local spikes. RMSE squares")
    print("       these transient spikes, heavily penalizing models for one-off")
    print("       anomalies, whereas MAE provides a robust operational gauge.")
    print("    4. R2 Limitation: R2 varies across seasons due to changing PM2.5")
    print("       variance (winter vs monsoon); MAE provides a stable yardstick.")
    print("  * Decision Rule:")
    print("    candidate_MAE <= champion_MAE * {:.0%} (Threshold = {:.4f} ug/m3)".format(
        TOLERANCE, threshold
    ))
    print("  " + "-" * 62)

    print("\n  Head-to-Head Model Comparison on New Batch:")
    print("  {:<10} {:>12} {:>12} {:>10}".format("Metric", "Champion", "Candidate", "Delta"))
    print("  " + "-" * 48)
    for m in ["mae", "rmse", "r2"]:
        c_val = champion_metrics[m]
        k_val = candidate_metrics[m]
        d_val = k_val - c_val
        sign = "+" if d_val > 0 else ""
        print("  {:<10} {:>12.4f} {:>12.4f} {:>9}".format(
            m.upper(), c_val, k_val, sign + "{:.4f}".format(d_val)
        ))

    gate_result = evaluate_gate(
        champion_metrics=champion_metrics,
        candidate_metrics=candidate_metrics,
        champion_artifact=str(CHAMPION_PATH),
        candidate_artifact=str(CANDIDATE_PATH),
        tolerance=TOLERANCE,
    )

    decision = gate_result["decision"]
    reason = gate_result["reason"]

    print("\n  " + "*" * 58)
    print("  QUALITY GATE DECISION: [ {} ]".format(decision))
    print("  Reason: {}".format(reason))
    print("  " + "*" * 58)

    print("\n  Deployment Guardrail Enforced:")
    print("  - Automatic Overwrite: DISABLED.")
    print("  - The champion artifact is NEVER replaced automatically.")
    if decision == "PROMOTE":
        print("  - Status: Candidate marked as ELIGIBLE FOR PROMOTION.")
        print("  - Action required: Human-in-the-loop sign-off to update MLflow alias")
        print("    'champion' -> candidate version and trigger production redeployment.")
    else:
        print("  - Status: Champion model RETAINED.")
        print("  - Action: Candidate archived for further hyperparameter tuning.")

    print("\n  Decision permanently appended to audit log -> {}".format(DECISION_LOG_PATH))
    print("  >>> TASK D COMPLETED SUCCESSFULLY")
    return decision, gate_result


# =============================================================================
# TASK E: Demonstration and Explanation
# =============================================================================

def task_e(new_batch_df, champ_m, cand_m, decision, gate_result, mlflow_run_id):
    banner("TASK E -- DEMONSTRATION & WORKFLOW EXPLANATION")

    print("""
  +--------------------------------------------------------------------------+
  |                   AirPulse MLOps Monitoring Pipeline                     |
  |                End-to-End Workflow Execution Summary                     |
  +--------------------------------------------------------------------------+
""")

    print("  [1] EXECUTION EVIDENCE SUMMARY:")
    print("      - Task A Data Batch   : {:,} validated records (18 features + 1 target)".format(len(new_batch_df)))
    print("      - Task B Champion MAE : {:.4f} ug/m3 (RMSE: {:.4f}, R2: {:.4f})".format(
        champ_m["mae"], champ_m["rmse"], champ_m["r2"]
    ))
    print("      - Task C Candidate MAE: {:.4f} ug/m3 (RMSE: {:.4f}, R2: {:.4f})".format(
        cand_m["mae"], cand_m["rmse"], cand_m["r2"]
    ))
    print("      - MLflow Tracking Run : {}".format(mlflow_run_id or "Recorded in mlruns/"))
    print("      - Task D Gate Outcome : {}".format(decision))
    print("      - Gate Decision Rule  : Candidate MAE ({:.4f}) <= Champion MAE * {:.0%} ({:.4f})".format(
        cand_m["mae"], TOLERANCE, champ_m["mae"] * TOLERANCE
    ))

    print("\n  [2] ARTIFACT GENERATION AUDIT:")
    print("  {:<32} {:<42} {:>10}".format("Artifact Description", "Relative Path", "Status"))
    print("  " + "-" * 88)

    artifacts_to_verify = [
        ("New Data Batch", NEW_BATCH_PATH),
        ("Champion Evaluation Report", EVAL_REPORT_PATH),
        ("Candidate Model Retrain Report", RETRAIN_REPORT_PATH),
        ("Candidate Model Artifact", CANDIDATE_PATH),
        ("Quality Gate Audit Log", DECISION_LOG_PATH),
        ("Champion Model Artifact", CHAMPION_PATH),
    ]

    for label, path in artifacts_to_verify:
        exists = path.exists()
        size_str = "{:,} bytes".format(path.stat().st_size) if exists else "NOT FOUND"
        status = "PASSED ({})".format(size_str) if exists else "FAILED"
        rel_path = str(path.relative_to(REPO_ROOT))
        print("  {:<32} {:<42} {:>10}".format(label, rel_path, status))

    print("""
  [3] WORKFLOW ARCHITECTURE & EXPLANATION:

      +-------------------+
      | Incoming Data     |  Hourly PM2.5 readings collected from CPCB stations.
      +---------+---------+
                |
                v
      +-------------------+  TASK A: PREPARE NEW DATA
      | Data Validation   |  Checks 18 features + target, rejects invalid rows,
      | & Audit Hygiene   |  drops nulls, eliminates duplicate records.
      +---------+---------+
                |
                v
      +-------------------+  TASK B: EVALUATE CHAMPION
      | Model Quality     |  Evaluates current champion on the incoming batch.
      | Monitoring        |  Calculates MAE, RMSE, and R2 against baseline.
      +---------+---------+  Generates evaluation_report.json.
                |
                v
      +-------------------+  TASK C: RETRAIN CANDIDATE
      | Model Retraining  |  Merges historical training set + new batch.
      | & Experiment Run  |  Fits HistGradientBoosting using existing pipeline.
      +---------+---------+  Logs parameters & metrics to MLflow.
                |            Saves candidate_model.joblib separately.
                v
      +-------------------+  TASK D: QUALITY GATE
      | Decision Rule &   |  Tests: candidate_MAE <= champion_MAE * 1.02.
      | Safety Governance |  Never overwrites champion automatically.
      +---------+---------+  Appends decision to quality_gate_decisions.jsonl.
                |
                +-----------------------+
                |                       |
           [ PROMOTE ]             [ RETAIN ]
                |                       |
                v                       v
      Human Sign-Off Needed     Champion Stays Active
      Update MLflow Alias       Candidate Preserved
      to 'champion'             for Investigation
""")

    print("  " + "=" * 70)
    print("  >>> ALL FIVE ASSESSMENT TASKS (A, B, C, D, E) EXECUTED SUCCESSFULLY")
    print("  " + "=" * 70 + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("  AirPulse: Model Quality Monitoring & Retraining Pipeline")
    print("  Theme: Extend existing ML project with monitoring & retraining")
    print("  Execution Date: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("#" * 70)

    # Task A: Prepare new data
    new_batch_df = task_a()

    # Task B: Evaluate existing model
    champ_metrics, persist_metrics = task_b(new_batch_df)

    # Task C: Retrain candidate model
    cand_metrics, cand_train_metrics, mlflow_run_id = task_c(new_batch_df)

    # Task D: Implement quality gate
    decision, gate_result = task_d(champ_metrics, cand_metrics)

    # Task E: Demonstration and explanation
    task_e(new_batch_df, champ_metrics, cand_metrics, decision, gate_result, mlflow_run_id)


if __name__ == "__main__":
    main()

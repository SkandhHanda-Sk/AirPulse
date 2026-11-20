"""
monitoring/quality_gate.py
--------------------------
Quality-gate logic for AirPulse model promotion.

Decision rule (regression — MAE is primary metric):
  * If candidate_mae  <=  champion_mae * TOLERANCE  -> PROMOTE candidate
  * Otherwise                                        -> RETAIN champion

Justification for MAE as gating metric
---------------------------------------
1. The HGB model is trained with loss="absolute_error" (MAE), making MAE the
   natural, internally consistent criterion.
2. MAE is directly interpretable: "predictions are off by X ug/m3 on average."
   Health-advisory thresholds are expressed in ug/m3, so the metric maps
   directly onto real-world impact.
3. Unlike RMSE, MAE is not inflated by occasional PM2.5 spikes — which are
   common in Delhi data and are not modelling failures per se.
4. R2 can be misleading when the target distribution shifts (e.g. a new
   season with different variance); MAE degrades more predictably.
"""
import json
import os
from datetime import datetime, timezone


# 2% tolerance: candidate must beat or match champion within 2%
TOLERANCE = 1.02

DECISION_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "reports", "quality_gate_decisions.jsonl"
)


def evaluate_gate(champion_metrics, candidate_metrics,
                  champion_artifact, candidate_artifact,
                  tolerance=TOLERANCE):
    """
    Compare champion vs candidate and return a gate result dict.

    Parameters
    ----------
    champion_metrics  : dict with keys 'mae', 'rmse', 'r2'
    candidate_metrics : dict with keys 'mae', 'rmse', 'r2'
    champion_artifact : path to current champion joblib
    candidate_artifact: path to candidate joblib
    tolerance         : multiplier applied to champion_mae (default 1.02)
    """
    champ_mae = float(champion_metrics["mae"])
    cand_mae  = float(candidate_metrics["mae"])
    threshold = champ_mae * tolerance

    if cand_mae <= threshold:
        decision = "PROMOTE"
        reason = (
            "Candidate MAE {:.4f} <= threshold {:.4f} "
            "({:.0%} of champion MAE {:.4f}). "
            "Candidate is eligible for promotion.".format(
                cand_mae, threshold, tolerance, champ_mae
            )
        )
    else:
        decision = "RETAIN"
        reason = (
            "Candidate MAE {:.4f} > threshold {:.4f} "
            "({:.0%} of champion MAE {:.4f}). "
            "Champion is retained.".format(
                cand_mae, threshold, tolerance, champ_mae
            )
        )

    result = {
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "champion_mae":     champ_mae,
        "candidate_mae":    cand_mae,
        "champion_rmse":    float(champion_metrics["rmse"]),
        "candidate_rmse":   float(candidate_metrics["rmse"]),
        "champion_r2":      float(champion_metrics["r2"]),
        "candidate_r2":     float(candidate_metrics["r2"]),
        "tolerance":        tolerance,
        "decision":         decision,
        "reason":           reason,
        "champion_artifact": champion_artifact,
        "candidate_artifact": candidate_artifact,
    }
    _persist_decision(result)
    return result


def _persist_decision(result):
    """Append the gate decision to the JSONL decision log (never overwrites)."""
    os.makedirs(os.path.dirname(DECISION_LOG_PATH), exist_ok=True)
    with open(DECISION_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")


def load_decision_log():
    """Return all recorded gate decisions as a list of dicts."""
    if not os.path.exists(DECISION_LOG_PATH):
        return []
    with open(DECISION_LOG_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

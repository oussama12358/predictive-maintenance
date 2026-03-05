"""
model/evaluate.py
-----------------
Evaluation utilities for the predictive maintenance classifier.

Key design decisions:
  - We optimize threshold on Precision-Recall curve, NOT at 0.5
  - In PdM, missing a failure (FN) is far more costly than a false alarm (FP)
  - F-beta with beta=2 weights recall 2x more than precision
  - Threshold is saved as an artifact so it can be updated without retraining
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    fbeta_score,
    average_precision_score,
)


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, beta: float = 2.0) -> float:
    """
    Find the decision threshold that maximizes F-beta score.

    Uses beta=2 by default — recall is weighted 2x more than precision.
    This reflects the industrial reality: missing a failure is much worse
    than triggering an unnecessary inspection.

    Args:
        y_true:  Ground truth binary labels
        y_proba: Predicted probabilities for positive class
        beta:    F-beta weight (default 2.0 → recall-focused)

    Returns:
        Optimal threshold as a float
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Avoid division by zero; precision_recall_curve returns len(thresholds)+1 points
    f_scores = []
    for p, r in zip(precisions[:-1], recalls[:-1]):
        if (beta**2 * p + r) == 0:
            f_scores.append(0.0)
        else:
            f_beta = (1 + beta**2) * (p * r) / (beta**2 * p + r)
            f_scores.append(f_beta)

    best_idx = np.argmax(f_scores)
    optimal_threshold = float(thresholds[best_idx])

    print(f"   Optimal threshold (F{beta}): {optimal_threshold:.4f}")
    print(f"   Precision at threshold     : {precisions[best_idx]:.4f}")
    print(f"   Recall at threshold        : {recalls[best_idx]:.4f}")
    print(f"   F{beta} at threshold        : {f_scores[best_idx]:.4f}")

    return optimal_threshold


def evaluate_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    dataset_name: str = "Validation",
) -> dict:
    """
    Full evaluation suite for the trained model.

    Args:
        y_true:       Ground truth labels
        y_proba:      Predicted probabilities
        threshold:    Decision threshold (from find_optimal_threshold)
        dataset_name: Label for print output (e.g. "Validation", "Test")

    Returns:
        Dict of all computed metrics
    """
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["No Failure", "Failure"])

    print(f"\n{'='*55}")
    print(f"  {dataset_name} Evaluation  (threshold={threshold:.4f})")
    print(f"{'='*55}")
    print(f"  ROC-AUC            : {roc_auc:.4f}")
    print(f"  Avg Precision (AP) : {avg_precision:.4f}")
    print(f"  F2 Score           : {f2:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print(f"\n{report}")

    return {
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "f2_score": f2,
        "threshold": threshold,
        "confusion_matrix": cm.tolist(),
    }


def save_threshold(threshold: float, output_path: str = "model/artifacts/threshold.json") -> None:
    """
    Persist the decision threshold as a JSON artifact.
    This allows ops teams to adjust the threshold without retraining.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "threshold": threshold,
        "description": "F2-optimized decision threshold. Raise to reduce false alarms. Lower to increase sensitivity.",
        "risk_levels": {
            "low":    [0.0, 0.35],
            "medium": [0.35, 0.65],
            "high":   [0.65, 1.0],
        },
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"   Threshold saved to: {path.resolve()}")


def load_threshold(path: str = "model/artifacts/threshold.json") -> float:
    """Load the decision threshold artifact."""
    with open(path, "r") as f:
        data = json.load(f)
    return float(data["threshold"])
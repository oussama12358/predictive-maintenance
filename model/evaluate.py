"""
model/evaluate.py
-----------------
Evaluation utilities for the predictive maintenance classifier.

Key design decisions:
  - Threshold optimized on Precision-Recall curve (F2-score, recall-focused)
  - Threshold is ROUNDED to 2 decimal places max to avoid overfitting on test set
    (as recommended by Bastien: prefer 0.20 or 0.25 over 0.2241)
  - Bootstrap robustness check: evaluate threshold stability across N random
    subsets of the test set — if std is low, the threshold generalizes well
  - Threshold saved as artifact for ops-level adjustment without retraining
"""

import json
import numpy as np
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
    Find the decision threshold that maximizes F-beta score,
    then round it to 2 decimal places to avoid overfitting on the validation set.

    Bastien's remark: a threshold like 0.2241 is suspiciously precise —
    it likely overfits the specific validation split. Rounding to 0.22 or
    choosing the nearest "round" candidate (0.20, 0.25) gives a threshold
    that generalizes better when the data distribution shifts slightly.

    Args:
        y_true:  Ground truth binary labels
        y_proba: Predicted probabilities for positive class
        beta:    F-beta weight (default 2.0 — recall weighted 2x over precision)

    Returns:
        Rounded optimal threshold as a float
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    f_scores = []
    for p, r in zip(precisions[:-1], recalls[:-1]):
        denom = (beta**2 * p + r)
        if denom == 0:
            f_scores.append(0.0)
        else:
            f_scores.append((1 + beta**2) * (p * r) / denom)

    best_idx = np.argmax(f_scores)
    raw_threshold = float(thresholds[best_idx])

    # ── Round to 2 decimal places ─────────────────────────────────────────────
    # Avoids overfitting a hyper-precise value to the validation split.
    # e.g.  0.2241 → 0.22   |   0.3187 → 0.32   |   0.1956 → 0.20
    rounded_threshold = round(raw_threshold, 2)

    print(f"   Raw threshold (F{beta} argmax) : {raw_threshold:.4f}")
    print(f"   Rounded threshold (2 d.p.)   : {rounded_threshold:.2f}  ← used for inference")
    print(f"   Precision at raw threshold   : {precisions[best_idx]:.4f}")
    print(f"   Recall at raw threshold      : {recalls[best_idx]:.4f}")
    print(f"   F{beta} at raw threshold      : {f_scores[best_idx]:.4f}")

    # Show F2 at the rounded threshold too (should be very close)
    y_pred_rounded = (y_proba >= rounded_threshold).astype(int)
    f2_rounded = fbeta_score(y_true, y_pred_rounded, beta=beta)
    print(f"   F{beta} at rounded threshold  : {f2_rounded:.4f}  (delta: {abs(f_scores[best_idx] - f2_rounded):.4f})")

    return rounded_threshold


def bootstrap_threshold_robustness(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    n_iterations: int = 200,
    subsample_ratio: float = 0.70,
    beta: float = 2.0,
    random_seed: int = 42,
) -> dict:
    """
    Evaluate threshold robustness by measuring F-beta score stability
    across N random subsamples of the test set.

    This directly addresses Bastien's suggestion:
    "Evaluate robustness on a subset of your test — large enough to be
    representative — and check the value you get."

    A robust threshold shows:
      - Low std across bootstrap iterations (< 0.03)
      - Mean F-beta close to the full-set F-beta
      - Narrow 95% confidence interval

    Args:
        y_true:          Ground truth labels (test set)
        y_proba:         Predicted probabilities (test set)
        threshold:       The rounded threshold to evaluate
        n_iterations:    Number of bootstrap samples (default 200)
        subsample_ratio: Fraction of test set per sample (default 70%)
        beta:            F-beta weight
        random_seed:     For reproducibility

    Returns:
        Dict with mean, std, min, max, ci_95_low, ci_95_high, verdict
    """
    rng = np.random.default_rng(random_seed)
    n = len(y_true)
    sample_size = int(n * subsample_ratio)

    f_scores = []

    for _ in range(n_iterations):
        indices = rng.choice(n, size=sample_size, replace=False)
        y_sub   = y_true[indices]
        p_sub   = y_proba[indices]

        # Skip degenerate samples with no positives
        if y_sub.sum() == 0:
            continue

        y_pred = (p_sub >= threshold).astype(int)
        f2 = fbeta_score(y_sub, y_pred, beta=beta, zero_division=0)
        f_scores.append(f2)

    f_scores = np.array(f_scores)
    ci_low  = float(np.percentile(f_scores, 2.5))
    ci_high = float(np.percentile(f_scores, 97.5))
    std     = float(f_scores.std())

    # Verdict: std < 0.03 = stable, 0.03–0.06 = acceptable, > 0.06 = unstable
    if std < 0.03:
        verdict = "STABLE — threshold generalizes well"
    elif std < 0.06:
        verdict = "ACCEPTABLE — minor sensitivity to data distribution"
    else:
        verdict = "UNSTABLE — consider adjusting threshold or collecting more data"

    print(f"\n   Bootstrap Robustness ({n_iterations} iterations, {subsample_ratio:.0%} subsample):")
    print(f"   F{beta} mean  : {f_scores.mean():.4f}")
    print(f"   F{beta} std   : {std:.4f}")
    print(f"   F{beta} min   : {f_scores.min():.4f}")
    print(f"   F{beta} max   : {f_scores.max():.4f}")
    print(f"   95% CI        : [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   Verdict       : {verdict}")

    return {
        "n_iterations":    n_iterations,
        "subsample_ratio": subsample_ratio,
        "threshold":       threshold,
        "f2_mean":         round(float(f_scores.mean()), 4),
        "f2_std":          round(std, 4),
        "f2_min":          round(float(f_scores.min()), 4),
        "f2_max":          round(float(f_scores.max()), 4),
        "ci_95_low":       round(ci_low, 4),
        "ci_95_high":      round(ci_high, 4),
        "verdict":         verdict,
    }


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
        threshold:    Decision threshold (rounded, from find_optimal_threshold)
        dataset_name: Label for print output

    Returns:
        Dict of all computed metrics
    """
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc       = roc_auc_score(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    f2            = fbeta_score(y_true, y_pred, beta=2)
    cm            = confusion_matrix(y_true, y_pred)
    report        = classification_report(y_true, y_pred, target_names=["No Failure", "Failure"])

    print(f"\n{'='*55}")
    print(f"  {dataset_name} Evaluation  (threshold={threshold:.2f})")
    print(f"{'='*55}")
    print(f"  ROC-AUC            : {roc_auc:.4f}")
    print(f"  Avg Precision (AP) : {avg_precision:.4f}")
    print(f"  F2 Score           : {f2:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print(f"\n{report}")

    return {
        "roc_auc":          roc_auc,
        "avg_precision":    avg_precision,
        "f2_score":         f2,
        "threshold":        threshold,
        "confusion_matrix": cm.tolist(),
    }


def save_threshold(threshold: float, robustness: dict = None,
                   output_path: str = "model/artifacts/threshold.json") -> None:
    """
    Persist the decision threshold + robustness report as a JSON artifact.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "threshold":   threshold,
        "description": (
            f"F2-optimized threshold rounded to 2 decimal places. "
            f"Raw optimal was more precise but rounded to avoid overfitting "
            f"to a specific validation split. "
            f"Raise to reduce false alarms. Lower to increase sensitivity."
        ),
        "risk_levels": {
            "low":    [0.0,  0.35],
            "medium": [0.35, 0.65],
            "high":   [0.65, 1.0],
        },
        "robustness": robustness or {},
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"   Threshold saved to: {path.resolve()}")


def load_threshold(path: str = "model/artifacts/threshold.json") -> float:
    """Load the decision threshold artifact."""
    with open(path, "r") as f:
        data = json.load(f)
    return float(data["threshold"])
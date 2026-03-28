"""
model/train.py
--------------
End-to-end training pipeline for the AI4I 2020 Predictive Maintenance Dataset.

Dataset : AI4I 2020 — Stephan Matzka, HTW Berlin, Germany
DOI     : https://doi.org/10.24432/C5HS5C

Pipeline stages:
  1. Load AI4I 2020 data (UCI or CSV)
  2. Feature engineering via shared features.py
  3. Train/validation/test split — stratified, no leakage
  4. Handle class imbalance with SMOTE (train set only)
  5. Scale features (fit on train set only)
  6. Train XGBoost classifier with cross-validation
  7. Find optimal F2 threshold on validation set
  8. Full evaluation suite
  9. Save model, scaler, threshold artifacts

Run:
    python -m model.train
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from data.Load_ai4i import load_from_csv, clean_and_standardize
from model.features import engineer_features, get_feature_columns, get_target_column
from model.evaluate import find_optimal_threshold, evaluate_model, save_threshold, bootstrap_threshold_robustness


# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH        = "data/raw/ai4i2020.csv"
MODEL_ARTIFACT   = "model/artifacts/model.pkl"
SCALER_ARTIFACT  = "model/artifacts/scaler.pkl"
THRESHOLD_ARTIFACT = "model/artifacts/threshold.json"

RANDOM_SEED      = 42
TEST_SIZE        = 0.20   # 80/20 split
VAL_SIZE         = 0.15   # 15% of full data used as validation (from training split)

# XGBoost hyperparameters — tuned for imbalanced binary classification
# Note: use_label_encoder removed (deprecated in XGBoost 3.x)
# Note: n_jobs=1 for cross_val_score to avoid joblib serialization warnings on Windows
XGBOOST_PARAMS = {
    "n_estimators":     400,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": 1,
    "eval_metric":      "aucpr",
    "objective":        "binary:logistic",   # explicit — ensures sklearn sees classifier
    "random_state":     RANDOM_SEED,
    "n_jobs":           1,
    "tree_method":      "hist",
}


def load_data(path: str) -> pd.DataFrame:
    """Load and clean AI4I 2020 dataset."""
    df_raw = load_from_csv(path)
    df = clean_and_standardize(df_raw)
    print(f"✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Failure rate: {df[get_target_column()].mean():.1%}")
    return df


def run_training_pipeline() -> None:
    """
    Execute the full training pipeline from raw data to saved artifacts.
    """
    print("\n" + "="*55)
    print("  PREDICTIVE MAINTENANCE — TRAINING PIPELINE")
    print("="*55)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    df = load_data(DATA_PATH)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n[2/7] Engineering features...")
    df = engineer_features(df)
    feature_cols = get_feature_columns()
    target_col   = get_target_column()

    X = df[feature_cols]
    y = df[target_col]

    print(f"   Features used: {len(feature_cols)}")
    print(f"   Feature list : {feature_cols}")

    # ── 3. Train / validation / test split ────────────────────────────────────
    # IMPORTANT: Stratify on y to preserve class ratio in each split.
    # No temporal leakage risk here (cross-sectional data, not time series).
    print("\n[3/7] Splitting data (stratified)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=y_temp, random_state=RANDOM_SEED
    )

    print(f"   Train   : {len(X_train):,} rows  (failure rate: {y_train.mean():.1%})")
    print(f"   Val     : {len(X_val):,} rows  (failure rate: {y_val.mean():.1%})")
    print(f"   Test    : {len(X_test):,} rows  (failure rate: {y_test.mean():.1%})")

    # ── 4. Handle class imbalance with SMOTE ──────────────────────────────────
    # SMOTE is applied ONLY to the training set.
    # Applying it before splitting would cause data leakage into validation.
    print("\n[4/7] Applying SMOTE to training set...")
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"   Before SMOTE: {len(X_train):,} rows  "
          f"({y_train.sum()} positives, {(~y_train.astype(bool)).sum()} negatives)")
    print(f"   After SMOTE : {len(X_train_resampled):,} rows  "
          f"({y_train_resampled.sum()} positives, "
          f"{(~y_train_resampled.astype(bool)).sum()} negatives)")

    # ── 5. Feature scaling ────────────────────────────────────────────────────
    # Fit scaler on training data only. Transform val and test.
    # XGBoost is tree-based and doesn't require scaling, but we scale anyway
    # for API consistency and potential future model swaps (e.g., LightGBM, SVM).
    print("\n[5/7] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # ── 6. Cross-validation sanity check ─────────────────────────────────────
    print("\n[6/7] Training XGBoost classifier...")
    model = XGBClassifier(**XGBOOST_PARAMS)

    # Manual CV loop — fully compatible with sklearn 1.7.2 + XGBoost 3.x
    # cross_val_score + make_scorer API changed in sklearn 1.7 / XGBoost 3.x
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train_resampled)):
        X_fold_train = X_train_scaled[train_idx]
        y_fold_train = y_train_resampled[train_idx] if hasattr(y_train_resampled, '__getitem__') \
                       else np.array(y_train_resampled)[train_idx]
        X_fold_val   = X_train_scaled[val_idx]
        y_fold_val   = y_train_resampled[val_idx] if hasattr(y_train_resampled, '__getitem__') \
                       else np.array(y_train_resampled)[val_idx]

        fold_model = XGBClassifier(**XGBOOST_PARAMS)
        fold_model.fit(X_fold_train, y_fold_train, verbose=False)
        y_proba = fold_model.predict_proba(X_fold_val)[:, 1]
        score = roc_auc_score(y_fold_val, y_proba)
        cv_scores.append(score)

    cv_scores = np.array(cv_scores)
    print(f"   CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit on full resampled training data
    model.fit(
        X_train_scaled, y_train_resampled,
        eval_set=[(X_val_scaled, y_val.values)],
        verbose=False,
    )
    print("   Training complete.")

    # ── 7. Threshold optimization + evaluation ────────────────────────────────
    print("\n[7/7] Evaluating and saving artifacts...")

    val_proba  = model.predict_proba(X_val_scaled)[:, 1]
    test_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n   Finding optimal threshold on validation set...")
    threshold = find_optimal_threshold(y_val.values, val_proba, beta=2.0)

    evaluate_model(y_val.values,  val_proba,  threshold, dataset_name="Validation")
    evaluate_model(y_test.values, test_proba, threshold, dataset_name="Test (Hold-out)")

    # ── Bootstrap robustness check ────────────────────────────────────────────
    # Addresses Bastien's remark: verify the rounded threshold is stable
    # across random subsets of the test set before saving it.
    print("\n   Running bootstrap robustness check on test set...")
    robustness = bootstrap_threshold_robustness(
        y_true          = y_test.values,
        y_proba         = test_proba,
        threshold       = threshold,
        n_iterations    = 200,
        subsample_ratio = 0.70,
        beta            = 2.0,
        random_seed     = RANDOM_SEED,
    )

    # ── Feature importance ────────────────────────────────────────────────────
    importances = model.feature_importances_
    feat_importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("\n   Top feature importances:")
    print(feat_importance_df.to_string(index=False))

    # ── Save artifacts ────────────────────────────────────────────────────────
    Path("model/artifacts").mkdir(parents=True, exist_ok=True)

    joblib.dump(model,  MODEL_ARTIFACT)
    joblib.dump(scaler, SCALER_ARTIFACT)
    save_threshold(threshold, robustness=robustness, output_path=THRESHOLD_ARTIFACT)

    print(f"\n   Model saved   : {MODEL_ARTIFACT}")
    print(f"   Scaler saved  : {SCALER_ARTIFACT}")
    print(f"   Threshold saved: {THRESHOLD_ARTIFACT}")
    print("\n✅ Pipeline complete. All artifacts saved.\n")


if __name__ == "__main__":
    run_training_pipeline()
"""
model.py
========
LightGBM pairwise binary classifier for entity matching.

Key design choices:
  - LightGBM (fast, handles class imbalance well via scale_pos_weight)
  - Early stopping on a held-out split (prevents overfitting)
  - Threshold tuning on validation set by maximizing F0.5
  - Cross-validation utility for reliable OOF estimates

Usage
-----
From train.py::

    from model import EntityMatchModel, predictions_from_proba, FEATURE_COLS

    clf = EntityMatchModel()
    clf.fit(X_train, y_train, X_val, y_val)
    clf.tune_threshold(X_val, y_val, s1_ids_val, cand_ids_val)
    clf.save("output/model.pkl")
"""

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from evaluation import macro_f05, entity_f05


# ---------------------------------------------------------------------------
# Feature column list — single source of truth shared with features.py
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "country_match",
    # Name features
    "name_exact",
    "name_levenshtein",
    "name_jaccard_token",
    "name_jaccard_char2",
    "name_jaccard_char3",
    "name_token_overlap",
    "name_tfidf_cosine",
    "name_len_diff",
    "name_first_token_match",
    # Address features
    "address_exact",
    "address_levenshtein",
    "address_jaccard_token",
    "address_jaccard_char2",
    "address_jaccard_char3",
    "address_token_overlap",
    "address_tfidf_cosine",
    "address_len_diff",
]

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class EntityMatchModel:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model: lgb.LGBMClassifier | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "EntityMatchModel":
        """
        Fit the LightGBM classifier.
        If X_val/y_val are provided, early stopping is applied.
        """
        # Handle class imbalance: positive pairs are rare
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        scale = neg / pos if pos > 0 else 1.0
        print(f"  Class balance: {pos} positives, {neg} negatives  (scale_pos_weight={scale:.1f})")

        params = {**LGBM_PARAMS, "scale_pos_weight": scale}
        self.model = lgb.LGBMClassifier(**params)

        fit_kwargs: dict = {
            "callbacks": [lgb.log_evaluation(200)],
        }

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val[FEATURE_COLS], y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(200),
            ]

        self.model.fit(X_train[FEATURE_COLS], y_train, **fit_kwargs)
        print(f"  Best iteration: {self.model.best_iteration_}")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Model not trained — call fit() first."
        return self.model.predict_proba(X[FEATURE_COLS])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    # ------------------------------------------------------------------
    # Threshold tuning
    # ------------------------------------------------------------------

    def tune_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        s1_ids_val: pd.Series,
        cand_ids_val: pd.Series,
        thresholds: np.ndarray | None = None,
    ) -> float:
        """
        Grid-search over thresholds to maximize macro F0.5 on validation set.
        Updates self.threshold in-place and returns best threshold.
        """
        if thresholds is None:
            thresholds = np.arange(0.30, 0.96, 0.05)

        proba = self.predict_proba(X_val)

        # Build ground truth from y_val labels
        gt: dict = {}
        for s1_id, cand_id, label in zip(s1_ids_val, cand_ids_val, y_val):
            if s1_id not in gt:
                gt[s1_id] = set()
            if label == 1:
                gt[s1_id].add(cand_id)

        best_thresh, best_f05 = self.threshold, -1.0
        print("\nThreshold tuning:")
        for thresh in thresholds:
            preds_map = predictions_from_proba(proba, s1_ids_val, cand_ids_val, float(thresh))
            score = macro_f05(preds_map, gt)
            marker = " <-- best" if score > best_f05 else ""
            print(f"  thresh={thresh:.2f}  F0.5={score:.4f}{marker}")
            if score > best_f05:
                best_f05 = score
                best_thresh = float(thresh)

        print(f"\nSelected threshold: {best_thresh:.2f}  (F0.5={best_f05:.4f})")
        self.threshold = best_thresh
        return best_thresh

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        assert self.model is not None
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": FEATURE_COLS, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "EntityMatchModel":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Utility: convert probability array -> predictions dict
# ---------------------------------------------------------------------------

def predictions_from_proba(
    proba: np.ndarray,
    s1_ids,
    cand_ids,
    threshold: float,
) -> dict:
    """Convert model output -> {s1_entity_id: set_of_matched_ids}."""
    result: dict = {}
    for prob, s1_id, cand_id in zip(proba, s1_ids, cand_ids):
        if s1_id not in result:
            result[s1_id] = set()
        if prob >= threshold:
            result[s1_id].add(cand_id)
    return result


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(features_df: pd.DataFrame, n_splits: int = 5) -> None:
    """
    Stratified K-fold cross-validation.
    features_df must contain columns: source1_entity_id, candidate_entity_id,
    label (0/1), and all FEATURE_COLS.
    """
    X = features_df[FEATURE_COLS]
    y = features_df["label"]
    s1_ids = features_df["source1_entity_id"]
    cand_ids = features_df["candidate_entity_id"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_scores = []

    print(f"\nRunning {n_splits}-fold CV...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        params = {**LGBM_PARAMS, "n_estimators": 500}
        clf = EntityMatchModel()
        clf.model = lgb.LGBMClassifier(**params)
        clf.model.fit(X_tr, y_tr)

        proba = clf.predict_proba(X_val)
        gt: dict = {}
        for sid, cid, label in zip(s1_ids.iloc[val_idx], cand_ids.iloc[val_idx], y_val):
            if sid not in gt:
                gt[sid] = set()
            if label == 1:
                gt[sid].add(cid)

        preds_map = predictions_from_proba(proba, s1_ids.iloc[val_idx], cand_ids.iloc[val_idx], 0.5)
        score = macro_f05(preds_map, gt)
        oof_scores.append(score)
        print(f"  Fold {fold + 1}/{n_splits}: F0.5={score:.4f}")

    print(f"\nMean CV F0.5 : {np.mean(oof_scores):.4f}")
    print(f"Std  CV F0.5 : {np.std(oof_scores):.4f}")

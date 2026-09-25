"""
train.py
========
Full training pipeline:
  1. Load training data (S1, S2, S3 + ground truth)
  2. Train/validation split (stratified by S1 entity)
  3. Blocking -> candidate pairs on train split
  4. Build pairwise labels from ground truth
  5. Feature extraction
  6. LightGBM training with early stopping
  7. Threshold tuning on validation set
  8. Save model + print validation F0.5

Usage
-----
From project root (dataset/ must exist):

    # Full training run
    python src/train.py

    # With explicit paths
    python src/train.py \
        --train-dir dataset/train \
        --output-dir output \
        --val-frac 0.2 \
        --cv
"""

import os
import sys
import argparse
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import preprocess_dataframe
from blocking import generate_candidates, save_candidate_pairs
from features import build_feature_matrix
from model import EntityMatchModel, predictions_from_proba, FEATURE_COLS, cross_validate_model
from evaluation import evaluate_on_validation, load_ground_truth, macro_f05, entity_f05


# ---------------------------------------------------------------------------
# Helper: build labeled feature DataFrame from candidate pairs + ground truth
# ---------------------------------------------------------------------------

def build_labeled_features(
    candidates_map: dict,
    gt: dict,
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flatten candidates_map to (s1_id, cand_id) pairs and label them 1/0
    using ground truth.
    """
    pairs = [
        (s1_id, cid)
        for s1_id, cids in candidates_map.items()
        for cid in cids
    ]
    print(f"  Building features for {len(pairs):,} candidate pairs...")
    feat_df = build_feature_matrix(pairs, s1_df, cand_df)

    if feat_df.empty:
        return feat_df

    # Label: 1 if candidate is in ground truth for that S1 entity
    def label_row(row):
        true_set = gt.get(row["source1_entity_id"], set())
        return 1 if row["candidate_entity_id"] in true_set else 0

    feat_df["label"] = feat_df.apply(label_row, axis=1)

    pos = feat_df["label"].sum()
    neg = len(feat_df) - pos
    print(f"  Labels: {pos} positive, {neg} negative  (ratio 1:{neg//max(pos,1)})")
    return feat_df


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    train_dir: str = "dataset/train",
    output_dir: str = "output",
    val_frac: float = 0.2,
    run_cv: bool = False,
    random_state: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(random_state)
    np.random.seed(random_state)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading training data...")
    s1 = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print(f"  S1: {len(s1)} rows  S2: {len(s2)} rows  S3: {len(s3)} rows")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print("\nPreprocessing...")
    s1 = preprocess_dataframe(s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    gt = load_ground_truth(gt_path)

    cand_all = pd.concat([s2, s3], ignore_index=True)

    # ------------------------------------------------------------------
    # 3. Train / Validation split (split on S1 entities)
    # ------------------------------------------------------------------
    print(f"\nSplitting S1 entities (val_frac={val_frac})...")
    s1_ids = s1["entity_id"].values
    n_val = max(1, int(len(s1_ids) * val_frac))
    rng = np.random.default_rng(random_state)
    val_idx = set(rng.choice(len(s1_ids), size=n_val, replace=False).tolist())

    s1_train = s1.iloc[[i for i in range(len(s1)) if i not in val_idx]].reset_index(drop=True)
    s1_val = s1.iloc[list(val_idx)].reset_index(drop=True)

    print(f"  Train S1: {len(s1_train)}  Val S1: {len(s1_val)}")

    # ------------------------------------------------------------------
    # 4. Blocking on training split
    # ------------------------------------------------------------------
    print("\nGenerating candidates (TRAIN split)...")
    cand_train = generate_candidates(s1_train, s2, s3)
    print("\nGenerating candidates (VAL split)...")
    cand_val = generate_candidates(s1_val, s2, s3)

    # ------------------------------------------------------------------
    # 5. Build labeled feature matrices
    # ------------------------------------------------------------------
    print("\nBuilding training feature matrix...")
    train_feat = build_labeled_features(cand_train, gt, s1_train, cand_all)

    print("\nBuilding validation feature matrix...")
    val_feat = build_labeled_features(cand_val, gt, s1_val, cand_all)

    if train_feat.empty:
        print("ERROR: No training features — check blocking strategy.")
        return

    # ------------------------------------------------------------------
    # 6. Optional Cross-Validation
    # ------------------------------------------------------------------
    if run_cv:
        print("\nRunning 5-fold cross-validation on TRAINING set...")
        cross_validate_model(train_feat, n_splits=5)

    # ------------------------------------------------------------------
    # 7. Train model
    # ------------------------------------------------------------------
    print("\nTraining LightGBM classifier...")
    X_train = train_feat[FEATURE_COLS]
    y_train = train_feat["label"]
    X_val = val_feat[FEATURE_COLS]
    y_val = val_feat["label"]

    clf = EntityMatchModel()
    clf.fit(X_train, y_train, X_val, y_val)

    # Print feature importance
    print("\nTop-10 feature importances:")
    fi = clf.feature_importance().head(10)
    for _, row in fi.iterrows():
        print(f"  {row['feature']:35s} {row['importance']:>6.0f}")

    # ------------------------------------------------------------------
    # 8. Threshold tuning on validation set
    # ------------------------------------------------------------------
    print("\nTuning decision threshold on validation set...")
    clf.tune_threshold(
        X_val, y_val,
        val_feat["source1_entity_id"],
        val_feat["candidate_entity_id"],
    )

    # ------------------------------------------------------------------
    # 9. Final validation F0.5 at tuned threshold
    # ------------------------------------------------------------------
    print("\nFinal validation evaluation:")
    proba_val = clf.predict_proba(X_val)
    preds_val = predictions_from_proba(
        proba_val,
        val_feat["source1_entity_id"],
        val_feat["candidate_entity_id"],
        clf.threshold,
    )
    # Add singletons (S1 entities with no candidates)
    for s1_id in s1_val["entity_id"]:
        if s1_id not in preds_val:
            preds_val[s1_id] = set()

    evaluate_on_validation(preds_val, gt_path)

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    model_path = os.path.join(output_dir, "model.pkl")
    clf.save(model_path)

    # Save candidate pairs (for submission zip)
    save_candidate_pairs(
        {**cand_train, **cand_val},
        os.path.join(output_dir, "train_candidate_pairs.tsv"),
    )

    print("\nTraining complete.")
    print(f"  Model        -> {model_path}")
    print(f"  Threshold    -> {clf.threshold:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train entity matching model")
    default_train = "data/dataset/train" if os.path.exists("data/dataset/train") else "dataset/train"
    parser.add_argument("--train-dir", default=default_train)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--cv", action="store_true", help="Run 5-fold CV")
    args = parser.parse_args()
    train(args.train_dir, args.output_dir, args.val_frac, args.cv)

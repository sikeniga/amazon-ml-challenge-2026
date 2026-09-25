"""
inference.py
============
End-to-end inference:
  1. Load test data (test_source1/2/3.tsv)
  2. Preprocess
  3. Blocking  -> candidate_pairs.tsv
  4. Feature extraction
  5. Load trained model, score pairs
  6. Write matching_results.tsv

Usage
-----
    python src/inference.py
    python src/inference.py --test-dir dataset/test --output-dir output \
        --model-path output/model.pkl --threshold 0.65
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import preprocess_dataframe
from blocking import generate_candidates, save_candidate_pairs
from features import build_feature_matrix
from model import EntityMatchModel, predictions_from_proba


def run_inference(
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    model_path: str = "output/model.pkl",
    threshold: float | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print("Loading test data...")
    s1 = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t")
    print(f"  S1: {len(s1)}  S2: {len(s2)}  S3: {len(s3)}")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print("Preprocessing...")
    s1 = preprocess_dataframe(s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    cand_all = pd.concat([s2, s3], ignore_index=True)

    # ------------------------------------------------------------------
    # 3. Blocking
    # ------------------------------------------------------------------
    print("Generating candidates (blocking)...")
    candidates_map = generate_candidates(s1, s2, s3)
    cand_pairs_path = os.path.join(output_dir, "candidate_pairs.tsv")
    save_candidate_pairs(candidates_map, cand_pairs_path)

    pairs = [
        (s1_id, cid)
        for s1_id, cids in candidates_map.items()
        for cid in cids
    ]
    print(f"Candidate pairs for scoring: {len(pairs):,}")

    # ------------------------------------------------------------------
    # 4. Handle zero candidates edge case
    # ------------------------------------------------------------------
    if not pairs:
        print("WARNING: No candidate pairs found — writing all-singleton prediction.")
        rows = [
            {"source1_entity_id": sid, "matched_entity_ids": ""}
            for sid in s1["entity_id"]
        ]
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "matching_results.tsv"), sep="\t", index=False
        )
        return

    # ------------------------------------------------------------------
    # 5. Feature extraction
    # ------------------------------------------------------------------
    print("Building feature matrix...")
    feat_df = build_feature_matrix(pairs, s1, cand_all)

    # ------------------------------------------------------------------
    # 6. Load model & predict
    # ------------------------------------------------------------------
    print(f"Loading model from {model_path}...")
    clf = EntityMatchModel.load(model_path)
    if threshold is not None:
        clf.threshold = threshold
        print(f"  Overriding threshold -> {threshold:.2f}")

    print(f"Scoring pairs (threshold={clf.threshold:.2f})...")
    proba = clf.predict_proba(feat_df)
    preds_map = predictions_from_proba(
        proba,
        feat_df["source1_entity_id"],
        feat_df["candidate_entity_id"],
        clf.threshold,
    )

    # Ensure every S1 entity appears in output (singletons -> empty)
    for s1_id in s1["entity_id"]:
        if s1_id not in preds_map:
            preds_map[s1_id] = set()

    # ------------------------------------------------------------------
    # 7. Write matching_results.tsv
    # ------------------------------------------------------------------
    results_path = os.path.join(output_dir, "matching_results.tsv")
    rows = [
        {
            "source1_entity_id": sid,
            "matched_entity_ids": ",".join(sorted(preds_map[sid])),
        }
        for sid in sorted(preds_map.keys())
    ]
    pd.DataFrame(rows).to_csv(results_path, sep="\t", index=False)

    n_matched = sum(1 for r in rows if r["matched_entity_ids"])
    n_singleton = len(rows) - n_matched
    print(f"\nWrote {results_path}")
    print(f"  Total S1 rows  : {len(rows)}")
    print(f"  Matched        : {n_matched}")
    print(f"  Singletons     : {n_singleton}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity matching inference")
    default_test = "data/dataset/test" if os.path.exists("data/dataset/test") else "dataset/test"
    parser.add_argument("--test-dir", default=default_test)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model-path", default="output/model.pkl")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override model threshold (default: use saved threshold)")
    args = parser.parse_args()
    run_inference(args.test_dir, args.output_dir, args.model_path, args.threshold)

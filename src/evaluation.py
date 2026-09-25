"""
evaluation.py
=============
F0.5 score computation — exactly matches the official leaderboard metric.

Formula:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Evaluation is MACRO-averaged: F0.5 is computed per Source-1 entity,
then averaged across all S1 entities.

Singleton rules:
  - Correct empty prediction  -> 1.0
  - Any prediction for a true singleton -> 0.0
"""

import statistics
import pandas as pd


def f05_score(precision: float, recall: float) -> float:
    """Core F0.5 formula (beta=0.5, beta^2=0.25)."""
    beta2 = 0.25
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def entity_f05(predicted: set, ground_truth: set) -> float:
    """F0.5 for a single S1 entity."""
    # Singleton cases
    if not ground_truth and not predicted:
        return 1.0   # correctly identified singleton
    if not ground_truth and predicted:
        return 0.0   # false merge on singleton

    if not predicted:
        # Recall = 0, F0.5 = 0
        return 0.0

    tp = len(predicted & ground_truth)
    precision = tp / len(predicted)
    recall = tp / len(ground_truth)
    return f05_score(precision, recall)


def macro_f05(predictions: dict, ground_truth: dict) -> float:
    """
    Macro-averaged F0.5 across all S1 entities present in ground_truth.

    Parameters
    ----------
    predictions   : {s1_entity_id: set_of_predicted_ids}
    ground_truth  : {s1_entity_id: set_of_true_ids}
    """
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        scores.append(entity_f05(pred_set, true_set))
    return sum(scores) / len(scores) if scores else 0.0


def load_ground_truth(path: str) -> dict:
    """Load ground truth TSV -> {s1_id: set_of_match_ids}."""
    df = pd.read_csv(path, sep="\t")
    result = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        if pd.isna(matched) or str(matched).strip() == "":
            result[s1_id] = set()
        else:
            result[s1_id] = set(str(matched).strip().split(","))
    return result


def load_predictions(path: str) -> dict:
    """Load prediction TSV -> {s1_id: set_of_match_ids}."""
    df = pd.read_csv(path, sep="\t")
    result = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        if pd.isna(matched) or str(matched).strip() == "":
            result[s1_id] = set()
        else:
            result[s1_id] = set(str(matched).strip().split(","))
    return result


def evaluate_on_validation(
    predictions: dict,
    ground_truth_path: str,
    verbose: bool = True,
) -> float:
    """
    Compute macro F0.5 against a ground-truth file.
    Prints a summary if verbose=True.
    """
    gt = load_ground_truth(ground_truth_path)
    entity_scores = [entity_f05(predictions.get(sid, set()), tset) for sid, tset in gt.items()]
    score = sum(entity_scores) / len(entity_scores) if entity_scores else 0.0

    if verbose:
        print(f"Macro F0.5  : {score:.4f}")
        print(f"  Entities  : {len(entity_scores)}")
        print(f"  Min       : {min(entity_scores):.4f}")
        print(f"  Max       : {max(entity_scores):.4f}")
        print(f"  Median    : {statistics.median(entity_scores):.4f}")
        n_singletons = sum(1 for sid, tset in gt.items() if not tset)
        correct_singletons = sum(
            1 for sid, tset in gt.items() if not tset and not predictions.get(sid, set())
        )
        print(f"  Singletons: {n_singletons} total, {correct_singletons} correctly predicted")
    return score

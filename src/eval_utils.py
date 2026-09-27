import numpy as np
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Any

def evaluate_entity_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: List[str]
) -> Dict[str, Any]:
    """
    Computes exact entity-level Macro F0.5 as required by Amazon ML Challenge 2026.
    
    For every Source-1 entity:
        true_matches = ground_truth[s1_id]
        predicted_matches = predictions[s1_id]
        
    Handles singletons (zero-match entities) rigorously:
        - True singleton & Predicted singleton: F0.5 = 1.0
        - True singleton & Predicted matches:   F0.5 = 0.0 (False match on singleton)
        - True matches & Predicted empty:       F0.5 = 0.0 (Missed all matches)
        - True matches & Predicted matches:     F0.5 = 1.25 * P * R / (0.25 * P + R)
        
    Macro F0.5 = mean(F0.5 over ALL Source-1 entities).
    """
    f05_scores = []
    entity_precisions = []
    entity_recalls = []
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    true_singletons = 0
    predicted_singletons = 0
    false_match_on_singletons = 0
    correctly_rejected_singletons = 0
    
    for sid in all_s1_ids:
        true_set = ground_truth.get(sid, set())
        pred_set = predictions.get(sid, set())
        
        is_true_singleton = (len(true_set) == 0)
        is_pred_singleton = (len(pred_set) == 0)
        
        if is_true_singleton:
            true_singletons += 1
            if is_pred_singleton:
                predicted_singletons += 1
                correctly_rejected_singletons += 1
                f05_scores.append(1.0)
                entity_precisions.append(1.0)
                entity_recalls.append(1.0)
            else:
                false_match_on_singletons += 1
                total_fp += len(pred_set)
                f05_scores.append(0.0)
                entity_precisions.append(0.0)
                entity_recalls.append(1.0)
        else:
            if is_pred_singleton:
                predicted_singletons += 1
                total_fn += len(true_set)
                f05_scores.append(0.0)
                entity_precisions.append(1.0)
                entity_recalls.append(0.0)
            else:
                tp = len(pred_set & true_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                
                p = tp / len(pred_set)
                r = tp / len(true_set)
                
                entity_precisions.append(p)
                entity_recalls.append(r)
                
                denom = 0.25 * p + r
                if denom > 0:
                    f05 = 1.25 * p * r / denom
                else:
                    f05 = 0.0
                f05_scores.append(f05)
                
    pair_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    pair_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_precision = float(np.mean(entity_precisions)) if entity_precisions else 0.0
    macro_recall = float(np.mean(entity_recalls)) if entity_recalls else 0.0
    
    return {
        'macro_f05': macro_f05,
        'pair_precision': pair_precision,
        'pair_recall': pair_recall,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'number_of_singletons': true_singletons,
        'number_of_predicted_singletons': predicted_singletons,
        'false_positive_count': total_fp,
        'false_negative_count': total_fn,
        'false_match_on_singletons': false_match_on_singletons,
        'correctly_rejected_singletons': correctly_rejected_singletons,
    }

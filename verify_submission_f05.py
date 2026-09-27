import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from src.eval_utils import evaluate_entity_macro_f05

def main():
    # 1. Load ground truth and validation split
    gt_df = pd.read_csv('data/dataset/train/train_ground_truth.tsv', sep='\t', nrows=10000)
    ground_truth = {}
    true_singletons = set()
    for _, r in gt_df.iterrows():
        sid = r['source1_entity_id']
        matches = set()
        if pd.notna(r['matched_entity_ids']):
            for mid in str(r['matched_entity_ids']).split(','):
                mid = mid.strip()
                if mid: matches.add(mid)
        ground_truth[sid] = matches
        if not matches:
            true_singletons.add(sid)

    val_s1_ids = list(gt_df['source1_entity_id'])

    # 2. Load cached validation candidates
    with open('output/val_cache.pkl', 'rb') as f:
        data = pickle.load(f)
    cand_records = data['cand_records']
    probs = data['probs']

    sid_src_cands = defaultdict(lambda: {'S2': [], 'S3': []})
    for i, (sid, cid, src, is_tp, s1_snum, csnum, s1_n, cn, s1_a, ca, score) in enumerate(cand_records):
        sid_src_cands[sid][src].append((cid, probs[i], score))

    def evaluate_config(cap, thresh):
        preds = {}
        for sid in val_s1_ids:
            s2_pool = sorted(sid_src_cands[sid]['S2'], key=lambda x: -x[2])[:cap]
            s3_pool = sorted(sid_src_cands[sid]['S3'], key=lambda x: -x[2])[:cap]
            s2_m = [c for c, p, _ in sorted(s2_pool, key=lambda x: -x[1]) if p >= thresh][:3]
            s3_m = [c for c, p, _ in sorted(s3_pool, key=lambda x: -x[1]) if p >= thresh][:3]
            preds[sid] = set(s2_m + s3_m)

        res = evaluate_entity_macro_f05(preds, ground_truth, val_s1_ids)

        # Per-entity breakdown
        single_scores = []
        nonsingle_scores = []
        for sid in val_s1_ids:
            t_set = ground_truth[sid]
            p_set = preds[sid]
            if not t_set:
                single_scores.append(1.0 if not p_set else 0.0)
            else:
                if not p_set:
                    nonsingle_scores.append(0.0)
                else:
                    tp = len(t_set & p_set)
                    p = tp / len(p_set)
                    r = tp / len(t_set)
                    denom = 0.25 * p + r
                    nonsingle_scores.append(1.25 * p * r / denom if denom > 0 else 0.0)

        res['singleton_acc'] = np.mean(single_scores)
        res['nonsingle_f05'] = np.mean(nonsingle_scores)
        return res

    # Evaluate configurations
    res_prev = evaluate_config(cap=6, thresh=0.87)
    res_curr = evaluate_config(cap=25, thresh=0.88)
    res_t87  = evaluate_config(cap=25, thresh=0.87)
    res_t89  = evaluate_config(cap=25, thresh=0.89)
    res_t90  = evaluate_config(cap=25, thresh=0.90)

    print("=" * 88)
    print("           DETAILED F0.5 BENCHMARK REPORT (10,000 Validation Entities)")
    print("=" * 88)
    print(f"{'Metric':<35} | {'Previous (Cap=6, T=0.87)':<24} | {'NEW OUTPUT (Cap=25, T=0.88)':<26}")
    print("-" * 88)
    print(f"{'Leaderboard Status':<35} | {'0.628 (Evaluated)':<24} | {'READY TO SUBMIT':<26}")
    print(f"{'Local Validation Macro F0.5':<35} | {res_prev['macro_f05']:<24.4f} | {res_curr['macro_f05']:<26.4f} (+{res_curr['macro_f05']-res_prev['macro_f05']:.4f})")
    print(f"{'Pair Precision':<35} | {res_prev['pair_precision']*100:<23.2f}% | {res_curr['pair_precision']*100:<25.2f}% (+{res_curr['pair_precision']*100-res_prev['pair_precision']*100:.2f}%)")
    print(f"{'Pair Recall':<35} | {res_prev['pair_recall']*100:<23.2f}% | {res_curr['pair_recall']*100:<25.2f}% (+{res_curr['pair_recall']*100-res_prev['pair_recall']*100:.2f}%)")
    print(f"{'False Positive Pairs':<35} | {res_prev['false_positive_count']:<24,d} | {res_curr['false_positive_count']:<26,d} (+{res_curr['false_positive_count']-res_prev['false_positive_count']:,d})")
    print(f"{'False Negative Pairs':<35} | {res_prev['false_negative_count']:<24,d} | {res_curr['false_negative_count']:<26,d} ({res_curr['false_negative_count']-res_prev['false_negative_count']:,d})")
    print("-" * 88)
    print("SINGLETON ENTITIES PERFORMANCE (557 Total Singletons):")
    print(f"{'Correctly Rejected (F0.5 = 1.0)':<35} | {res_prev['correctly_rejected_singletons']:<24,d} ({res_prev['singleton_acc']*100:.1f}%) | {res_curr['correctly_rejected_singletons']:<26,d} ({res_curr['singleton_acc']*100:.1f}%)")
    print(f"{'False Matches (F0.5 = 0.0)':<35} | {res_prev['false_match_on_singletons']:<24,d} | {res_curr['false_match_on_singletons']:<26,d}")
    print("-" * 88)
    print("NON-SINGLETON ENTITIES PERFORMANCE (9,443 Entities with Matches):")
    print(f"{'Average Entity F0.5':<35} | {res_prev['nonsingle_f05']:<24.4f} | {res_curr['nonsingle_f05']:<26.4f} (+{res_curr['nonsingle_f05']-res_prev['nonsingle_f05']:.4f})")
    print("=" * 88)

    print("\nEXPECTED LEADERBOARD SCORE UNDER DIFFERENT SINGLETON RATIOS:")
    print("-" * 75)
    print(f"{'Singleton % in Test Set':<25} | {'Previous (Cap=6)':<22} | {'New Output (Cap=25)':<22}")
    print("-" * 75)
    for p in [0.056, 0.15, 0.25, 0.35, 0.50]:
        prev_proj = p * res_prev['singleton_acc'] + (1 - p) * res_prev['nonsingle_f05']
        curr_proj = p * res_curr['singleton_acc'] + (1 - p) * res_curr['nonsingle_f05']
        print(f"{p*100:5.1f}% Singletons           | {prev_proj:22.4f} | {curr_proj:22.4f} (+{curr_proj-prev_proj:.4f})")
    print("-" * 75)

if __name__ == '__main__':
    main()

import os
import sys
import re
import pickle
from collections import defaultdict
import numpy as np

sys.path.append('.')
from src.eval_utils import evaluate_entity_macro_f05
from src.data_split import get_or_create_val_split
from rapidfuzz.distance import JaroWinkler
import rapidfuzz.fuzz as rfuzz

def get_all_nums(addr: str) -> set:
    if not isinstance(addr, str) or not addr: return set()
    return set(re.findall(r'\b\d+\b', addr))

def main():
    val_s1_ids = get_or_create_val_split(n_val=10000)
    with open('output/val_cache.pkl', 'rb') as f:
        data = pickle.load(f)
    cand_records = data['cand_records']
    probs = data['probs']

    import pandas as pd
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

    print(f"Loaded {len(probs):,} cached candidate predictions.")

    # Pre-extract all numbers for each record to test set intersection
    s1_all_nums = {}
    c_all_nums = {}
    for i, (sid, cid, src, is_tp, s1_snum, csnum, s1_n, cn, s1_a, ca, score) in enumerate(cand_records):
        if sid not in s1_all_nums:
            s1_all_nums[sid] = get_all_nums(s1_a)
        if cid not in c_all_nums:
            c_all_nums[cid] = get_all_nums(ca)

    experiments = []
    # Test 1: Number set disjointness (reject ONLY if both have numbers and intersection is completely empty)
    for thresh in [0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92]:
        experiments.append((f"Thresh {thresh:.2f} (Baseline)", thresh, 'none'))
        experiments.append((f"Thresh {thresh:.2f} + Disjoint Nums Filter", thresh, 'disjoint_nums'))
        experiments.append((f"Thresh {thresh:.2f} + Disjoint Nums (len>=2)", thresh, 'disjoint_long_nums'))

    print("\n" + "=" * 95)
    print(f"{'Experiment':<52} {'F0.5':<8} {'Prec':<8} {'Rec':<8} {'FP':<6} {'FN':<6} {'S_FP':<6}")
    print("=" * 95)

    best_f05 = 0.0
    best_name = None

    for name, thresh, rule in experiments:
        sid_cands = defaultdict(lambda: {'S2': [], 'S3': []})
        for i, (sid, cid, src, is_tp, s1_snum, csnum, s1_n, cn, s1_a, ca, score) in enumerate(cand_records):
            p = probs[i]
            if p < thresh: continue

            if rule == 'disjoint_nums':
                nums1 = s1_all_nums[sid]
                numsc = c_all_nums[cid]
                # If both have numbers and share NONE:
                if len(nums1) > 0 and len(numsc) > 0 and not (nums1 & numsc):
                    # But don't reject if names are extremely identical (e.g. jw_n > 0.98)
                    jw_n = JaroWinkler.similarity(s1_n, cn)
                    if jw_n < 0.95:
                        continue

            elif rule == 'disjoint_long_nums':
                # Only check numbers with length >= 2 (e.g. house numbers like 2898, 9906, not single-digit 1 or 2)
                nums1 = {x for x in s1_all_nums[sid] if len(x) >= 2}
                numsc = {x for x in c_all_nums[cid] if len(x) >= 2}
                if len(nums1) > 0 and len(numsc) > 0 and not (nums1 & numsc):
                    jw_n = JaroWinkler.similarity(s1_n, cn)
                    if jw_n < 0.95:
                        continue

            sid_cands[sid][src].append((cid, p))

        preds = {}
        for sid in val_s1_ids:
            s2m = [c for c, _ in sorted(sid_cands[sid]['S2'], key=lambda x: -x[1])][:3]
            s3m = [c for c, _ in sorted(sid_cands[sid]['S3'], key=lambda x: -x[1])][:3]
            preds[sid] = set(s2m + s3m)

        res = evaluate_entity_macro_f05(preds, ground_truth, val_s1_ids)
        s_fp = res['false_match_on_singletons']
        f05 = res['macro_f05']
        p = res['pair_precision']
        r = res['pair_recall']
        fp = res['false_positive_count']
        fn = res['false_negative_count']
        star = " *** NEW BEST ***" if f05 > best_f05 else ""
        if f05 > best_f05:
            best_f05 = f05
            best_name = name
        print(f"{name:<52} {f05:<8.4f} {p*100:<7.2f}% {r*100:<7.2f}% {fp:<6d} {fn:<6d} {s_fp:<6d}{star}")

    print("=" * 95)
    print(f"OVERALL BEST: {best_name} -> F0.5 = {best_f05:.4f}")

if __name__ == '__main__':
    main()

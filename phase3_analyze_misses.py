"""
phase3_threshold_sweep.py
==========================
Efficiently finds the optimal LightGBM threshold by:
1. Running the v3 blocking pipeline (cap=25, 10-tier, no exact shortcut)
2. Collecting ALL (prob, is_true_match) pairs
3. Sweeping thresholds [0.50 .. 0.95] to find peak macro F0.5
4. Reports the optimal threshold to use in generate_final_submission.py
"""

import os
import sys
import re
import json
import time
import pickle
import unicodedata
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
import rapidfuzz.fuzz as rfuzz

sys.path.append('.')
from src.eval_utils import evaluate_entity_macro_f05
from src.data_split import get_or_create_val_split

# ---- helpers (identical to generate_final_submission.py v3) ----

def clean_name(s):
    if not isinstance(s, str) or not s: return ''
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8').lower()
    for sep in [' d/b/a ', ' dba ', ' t/a ', ' ta ', ' trading as ']:
        if sep in s: s = s.split(sep)[-1]; break
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\b(inc|corp|corporation|incorporated|llc|pllc|ltd|limited|co|company|pvt|private|llp|pc|sarl|sas|sasu|sa|eurl|snc|sci|gie)\b', ' ', s)
    return ' '.join(s.split())

def squish(s):
    s = re.sub(r'\b(com|org|net|in|fr|io|co|biz|info)\b', '', s)
    return s.replace(' ', '')

def clean_addr(s):
    if not isinstance(s, str) or not s: return ''
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8').lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\b(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|way|suite|ste|apt|floor|fl)\b', ' ', s)
    return ' '.join(s.split())

def get_street_num(addr):
    if not isinstance(addr, str) or not addr: return ''
    nums = re.findall(r'\b\d+\b', addr)
    return nums[0] if nums else ''

def get_street_prefix(addr):
    if not isinstance(addr, str) or not addr: return ''
    ca = clean_name(addr)
    words = [w for w in ca.split() if not w.isdigit() and len(w) >= 3]
    return words[0][:3] if words else ''

def get_city_tokens(addr):
    """Last 3 long tokens from address (city/state at end of address string)."""
    if not isinstance(addr, str) or not addr: return []
    ca = clean_addr(addr)
    words = [w for w in ca.split() if not w.isdigit() and len(w) >= 4]
    return words[-3:] if len(words) >= 3 else words

def compute_features(s1_n, s1_sq, s1_a, s1_snum, s1_c, cn, csq, ca, csnum, cc, prio):
    jw_n = JaroWinkler.similarity(s1_n, cn)
    ts_n = rfuzz.token_sort_ratio(s1_n, cn) / 100.0
    tset_n = rfuzz.token_set_ratio(s1_n, cn) / 100.0
    pr_n = rfuzz.partial_ratio(s1_n, cn) / 100.0
    exact_n = float(s1_n == cn and s1_n != '')
    exact_sq = float(s1_sq == csq and len(s1_sq) >= 5)
    num_m = float(s1_snum == csnum and s1_snum != '')
    jw_a = JaroWinkler.similarity(s1_a, ca) if (s1_a and ca) else 0.0
    tset_a = rfuzz.token_set_ratio(s1_a, ca) / 100.0 if (s1_a and ca) else 0.0
    country_m = float(s1_c == cc)
    return [jw_n, ts_n, tset_n, pr_n, exact_n, exact_sq, num_m, jw_a, tset_a, country_m, float(prio)]

def f05_score(p, r):
    denom = 0.25 * p + r
    return 1.25 * p * r / denom if denom > 0 else 0.0

def main():
    print("=" * 75)
    print("  PHASE 3: THRESHOLD SWEEP — Find optimal LightGBM decision boundary")
    print("=" * 75)
    t0 = time.time()

    # 1. Load validation split + ground truth
    val_s1_ids = get_or_create_val_split(n_val=10000)
    val_set = set(val_s1_ids)

    gt_df = pd.read_csv('data/dataset/train/train_ground_truth.tsv', sep='\t', nrows=10000)
    ground_truth = {}
    total_true_matches = 0
    for _, r in gt_df.iterrows():
        sid = r['source1_entity_id']
        matches = set()
        if pd.notna(r['matched_entity_ids']):
            for mid in str(r['matched_entity_ids']).split(','):
                mid = mid.strip()
                if mid: matches.add(mid)
        ground_truth[sid] = matches
        total_true_matches += len(matches)

    print(f"Ground truth: {len(val_s1_ids):,} entities, {total_true_matches:,} true pairs")

    # 2. Load model
    with open('output/champion_model.pkl', 'rb') as f:
        bundle = pickle.load(f)
    clf = bundle['clf']
    print(f"Model loaded. Existing threshold in bundle: {bundle.get('threshold', 0.82):.3f}")

    # 3. Load S1 data
    s1_rows = []
    for chunk in pd.read_csv('data/dataset/train/train_source1.tsv', sep='\t', chunksize=200000):
        m = chunk[chunk['entity_id'].isin(val_set)]
        if len(m): s1_rows.append(m)
        if sum(len(x) for x in s1_rows) >= len(val_set): break
    s1_df = pd.concat(s1_rows).drop_duplicates('entity_id').set_index('entity_id')

    # 4. Build 10-tier S1 index
    idx_name = defaultdict(list); idx_sq = defaultdict(list); idx_fl = defaultdict(list)
    idx_w12 = defaultdict(list); idx_w23 = defaultdict(list); idx_tfp = defaultdict(list)
    idx_addr_num_st = defaultdict(list); idx_addr_num_name = defaultdict(list)
    idx_city_w12 = defaultdict(list); idx_num_city_w1 = defaultdict(list)
    s1_clean = {}

    for sid, r in s1_df.iterrows():
        c = str(r['country']).upper().strip()
        cn = clean_name(r['business_name']); sq = squish(cn)
        ca = clean_addr(r['business_address']); snum = get_street_num(r['business_address'])
        st_pref = get_street_prefix(r['business_address']); city_tok = get_city_tokens(r['business_address'])
        words = cn.split()
        s1_clean[sid] = (cn, sq, ca, snum, c)

        if cn: idx_name[(c, cn)].append(sid)
        if len(sq) >= 5: idx_sq[(c, sq)].append(sid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[-1]) >= 3: idx_fl[(c, words[0], words[-1])].append(sid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[1]) >= 3: idx_w12[(c, words[0], words[1])].append(sid)
        if len(words) >= 3 and len(words[1]) >= 3 and len(words[-1]) >= 3: idx_w23[(c, words[1], words[-1])].append(sid)
        if 2 <= len(words) <= 4: idx_tfp[(c, '|'.join(sorted(words)))].append(sid)
        if snum and st_pref: idx_addr_num_st[(c, snum, st_pref)].append(sid)
        if snum and len(words) >= 1 and len(words[0]) >= 3: idx_addr_num_name[(c, snum, words[0])].append(sid)
        if len(city_tok) >= 2: idx_city_w12[(c, city_tok[0], city_tok[1])].append(sid)
        if snum and len(city_tok) >= 1: idx_num_city_w1[(c, snum, city_tok[0])].append(sid)

    # Fan-out pruning: skip city keys mapping to >50 S1 entities (too generic = slow + noisy)
    MAX_FANOUT = 50
    idx_city_w12 = {k: v for k, v in idx_city_w12.items() if len(v) <= MAX_FANOUT}
    idx_num_city_w1 = {k: v for k, v in idx_num_city_w1.items() if len(v) <= MAX_FANOUT}

    # 5. Stream S2+S3, cap=25 per source
    s2_cands = defaultdict(dict); s3_cands = defaultdict(dict)
    MAX_CANDS = 25

    print("Streaming S2+S3 (cap=25)...")
    for fn in ['train_source2.tsv', 'train_source3.tsv']:
        is_s2 = 'source2' in fn
        target = s2_cands if is_s2 else s3_cands
        n = 0
        for chunk in pd.read_csv('data/dataset/train/' + fn, sep='\t', chunksize=400000):
            n += len(chunk)
            for cid, bname, baddr, bctry in zip(chunk['entity_id'], chunk['business_name'], chunk['business_address'], chunk['country']):
                c = str(bctry).upper().strip()
                cn = clean_name(bname); sq = squish(cn)
                ca = clean_addr(baddr); snum = get_street_num(baddr)
                st_pref = get_street_prefix(baddr); city_tok = get_city_tokens(baddr)
                words = cn.split()

                sid_scores = defaultdict(int)
                if cn and (c, cn) in idx_name:
                    for sid in idx_name[(c, cn)]: sid_scores[sid] = max(sid_scores[sid], 100)
                if len(sq) >= 5 and (c, sq) in idx_sq:
                    for sid in idx_sq[(c, sq)]: sid_scores[sid] = max(sid_scores[sid], 95)
                if 2 <= len(words) <= 4 and (c, '|'.join(sorted(words))) in idx_tfp:
                    for sid in idx_tfp[(c, '|'.join(sorted(words)))]: sid_scores[sid] = max(sid_scores[sid], 85)
                if len(words) >= 2 and len(words[0]) >= 3 and len(words[1]) >= 3 and (c, words[0], words[1]) in idx_w12:
                    for sid in idx_w12[(c, words[0], words[1])]: sid_scores[sid] = max(sid_scores[sid], 80)
                if len(words) >= 2 and len(words[0]) >= 3 and len(words[-1]) >= 3 and (c, words[0], words[-1]) in idx_fl:
                    for sid in idx_fl[(c, words[0], words[-1])]: sid_scores[sid] = max(sid_scores[sid], 75)
                if len(words) >= 3 and len(words[1]) >= 3 and len(words[-1]) >= 3 and (c, words[1], words[-1]) in idx_w23:
                    for sid in idx_w23[(c, words[1], words[-1])]: sid_scores[sid] = max(sid_scores[sid], 70)
                if snum and len(words) >= 1 and len(words[0]) >= 3 and (c, snum, words[0]) in idx_addr_num_name:
                    for sid in idx_addr_num_name[(c, snum, words[0])]: sid_scores[sid] = max(sid_scores[sid], 65)
                if snum and st_pref and (c, snum, st_pref) in idx_addr_num_st:
                    for sid in idx_addr_num_st[(c, snum, st_pref)]: sid_scores[sid] = max(sid_scores[sid], 60)
                if len(city_tok) >= 2 and (c, city_tok[0], city_tok[1]) in idx_city_w12:
                    for sid in idx_city_w12[(c, city_tok[0], city_tok[1])]: sid_scores[sid] = max(sid_scores[sid], 55)
                if snum and len(city_tok) >= 1 and (c, snum, city_tok[0]) in idx_num_city_w1:
                    for sid in idx_num_city_w1[(c, snum, city_tok[0])]: sid_scores[sid] = max(sid_scores[sid], 50)

                for sid, score in sid_scores.items():
                    cur = target[sid]
                    if len(cur) < MAX_CANDS:
                        cur[cid] = (score, cn, sq, ca, snum, c)
                    else:
                        min_cid = min(cur.keys(), key=lambda x: cur[x][0])
                        if score > cur[min_cid][0]:
                            del cur[min_cid]
                            cur[cid] = (score, cn, sq, ca, snum, c)
            print(f"  {fn}: {n:,} rows...", end='\r')
        print(f"  {fn}: {n:,} rows done.")

    # 6. Feature extraction + LightGBM scoring (collect ALL probs)
    print("Computing features and running LightGBM on all candidates...")
    batch_feats = []; batch_meta = []  # (sid, cid, src, is_true_match)

    for sid in val_s1_ids:
        if sid not in s1_clean: continue
        s1_n, s1_sq, s1_a, s1_snum, s1_c = s1_clean[sid]
        true_set = ground_truth.get(sid, set())

        for cid, (score, cn, csq, ca, csnum, cc) in s2_cands.get(sid, {}).items():
            batch_feats.append(compute_features(s1_n, s1_sq, s1_a, s1_snum, s1_c, cn, csq, ca, csnum, cc, score))
            batch_meta.append((sid, cid, 'S2', int(cid in true_set)))

        for cid, (score, cn, csq, ca, csnum, cc) in s3_cands.get(sid, {}).items():
            batch_feats.append(compute_features(s1_n, s1_sq, s1_a, s1_snum, s1_c, cn, csq, ca, csnum, cc, score))
            batch_meta.append((sid, cid, 'S3', int(cid in true_set)))

    X = np.array(batch_feats, dtype=np.float32)
    probs = clf.predict_proba(X)[:, 1]
    print(f"Scored {len(probs):,} candidate pairs.")

    # Probability distribution
    print(f"\nProbability distribution of ALL candidates:")
    for cutoff in [0.3, 0.5, 0.6, 0.7, 0.75, 0.80, 0.82, 0.85, 0.90, 0.95]:
        above = (probs >= cutoff).sum()
        print(f"  prob >= {cutoff:.2f}: {above:6,} pairs ({above/len(probs)*100:.1f}%)")

    # Distribution for TRUE pairs only
    true_probs = probs[[i for i, (_, _, _, tp) in enumerate(batch_meta) if tp == 1]]
    print(f"\nProbability distribution for TRUE pairs ({len(true_probs):,}):")
    for cutoff in [0.50, 0.60, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.90]:
        above = (true_probs >= cutoff).sum()
        print(f"  prob >= {cutoff:.2f}: {above:6,} / {len(true_probs):,} ({above/len(true_probs)*100:.1f}%)")

    # 7. Threshold sweep
    print("\n" + "=" * 75)
    print("  THRESHOLD SWEEP (F0.5 per threshold, cap=3 per source)")
    print("=" * 75)
    print(f"{'Threshold':>12} {'Precision':>12} {'Recall':>10} {'macro_F05':>12} {'FP':>8} {'FN':>8}")
    print("-" * 75)

    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.85, 0.87, 0.88, 0.89,
                  0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    best_f05 = 0.0; best_thresh = 0.82
    results = []

    for thresh in thresholds:
        # Build predictions at this threshold
        sid_src_cands = defaultdict(lambda: {'S2': [], 'S3': []})
        for i, (sid, cid, src, _) in enumerate(batch_meta):
            if probs[i] >= thresh:
                sid_src_cands[sid][src].append((cid, probs[i]))

        preds = {}
        for sid in val_s1_ids:
            s2m = [c for c, _ in sorted(sid_src_cands[sid]['S2'], key=lambda x: -x[1])][:3]
            s3m = [c for c, _ in sorted(sid_src_cands[sid]['S3'], key=lambda x: -x[1])][:3]
            preds[sid] = set(s2m + s3m)

        res = evaluate_entity_macro_f05(preds, ground_truth, val_s1_ids)
        f05 = res['macro_f05']
        prec = res['pair_precision']
        rec = res['pair_recall']
        fp = res['false_positive_count']
        fn = res['false_negative_count']

        results.append({'threshold': thresh, 'macro_f05': f05, 'precision': prec, 'recall': rec, 'fp': fp, 'fn': fn})
        marker = " <-- BEST" if f05 > best_f05 else ""
        print(f"  {thresh:>10.2f} {prec:>12.4f} {rec:>10.4f} {f05:>12.4f} {fp:>8,} {fn:>8,}{marker}")

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

    print("=" * 75)
    print(f"\n  OPTIMAL THRESHOLD: {best_thresh:.2f}  (macro_F0.5 = {best_f05:.4f})")
    print(f"\n  --> Update generate_final_submission.py threshold override to: {best_thresh:.2f}")

    # Save results
    with open('output/threshold_sweep.json', 'w') as f:
        json.dump({'best_threshold': best_thresh, 'best_f05': best_f05, 'results': results}, f, indent=2)
    print(f"  Results saved -> output/threshold_sweep.json")
    print(f"\n  Total runtime: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()

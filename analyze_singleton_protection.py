import os
import sys
import json
import time
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
import rapidfuzz.fuzz as rfuzz

sys.path.append('.')
from src.eval_utils import evaluate_entity_macro_f05
from src.data_split import get_or_create_val_split
from phase3_analyze_misses import clean_name, squish, clean_addr, get_street_num, get_street_prefix, get_city_tokens, compute_features

def main():
    t0 = time.time()
    cache_path = 'output/val_cache.pkl'

    val_s1_ids = get_or_create_val_split(n_val=10000)
    val_set = set(val_s1_ids)

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

    print(f"Validation: {len(val_s1_ids):,} entities, {len(true_singletons):,} singletons ({len(true_singletons)/len(val_s1_ids)*100:.2f}%)")

    if os.path.exists(cache_path):
        print(f"Loading cached predictions from {cache_path}...")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        cand_records = data['cand_records']
        probs = data['probs']
        print(f"Loaded {len(probs):,} cached candidate predictions in {time.time()-t0:.1f}s")
    else:
        # Load model
        with open('output/champion_model.pkl', 'rb') as f:
            clf = pickle.load(f)['clf']

        # Load S1
        s1_rows = []
        for chunk in pd.read_csv('data/dataset/train/train_source1.tsv', sep='\t', chunksize=200000):
            m = chunk[chunk['entity_id'].isin(val_set)]
            if len(m): s1_rows.append(m)
            if sum(len(x) for x in s1_rows) >= len(val_set): break
        s1_df = pd.concat(s1_rows).drop_duplicates('entity_id').set_index('entity_id')

        # Build 10-tier index
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
            s1_clean[sid] = (cn, sq, ca, snum, c, str(r['business_name']), str(r['business_address']))

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

        idx_city_w12 = {k: v for k, v in idx_city_w12.items() if len(v) <= 50}
        idx_num_city_w1 = {k: v for k, v in idx_num_city_w1.items() if len(v) <= 50}

        # Stream S2 & S3 (cap=25)
        s2_cands = defaultdict(dict); s3_cands = defaultdict(dict)
        print("Streaming S2+S3...")
        for fn in ['train_source2.tsv', 'train_source3.tsv']:
            is_s2 = 'source2' in fn
            target = s2_cands if is_s2 else s3_cands
            for chunk in pd.read_csv('data/dataset/train/' + fn, sep='\t', chunksize=400000):
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
                        if len(cur) < 25:
                            cur[cid] = (score, cn, sq, ca, snum, c, str(bname), str(baddr))
                        else:
                            min_cid = min(cur.keys(), key=lambda x: cur[x][0])
                            if score > cur[min_cid][0]:
                                del cur[min_cid]
                                cur[cid] = (score, cn, sq, ca, snum, c, str(bname), str(baddr))

        print("Scoring candidates...")
        batch_feats = []
        cand_records = []
        for sid in val_s1_ids:
            if sid not in s1_clean: continue
            s1_n, s1_sq, s1_a, s1_snum, s1_c, s1_raw_name, s1_raw_addr = s1_clean[sid]
            true_set = ground_truth.get(sid, set())

            for cid, (score, cn, csq, ca, csnum, cc, raw_n, raw_a) in s2_cands.get(sid, {}).items():
                f = compute_features(s1_n, s1_sq, s1_a, s1_snum, s1_c, cn, csq, ca, csnum, cc, score)
                batch_feats.append(f)
                cand_records.append((sid, cid, 'S2', int(cid in true_set), s1_snum, csnum, s1_n, cn, s1_a, ca, score))

            for cid, (score, cn, csq, ca, csnum, cc, raw_n, raw_a) in s3_cands.get(sid, {}).items():
                f = compute_features(s1_n, s1_sq, s1_a, s1_snum, s1_c, cn, csq, ca, csnum, cc, score)
                batch_feats.append(f)
                cand_records.append((sid, cid, 'S3', int(cid in true_set), s1_snum, csnum, s1_n, cn, s1_a, ca, score))

        X = np.array(batch_feats, dtype=np.float32)
        probs = clf.predict_proba(X)[:, 1]

        os.makedirs('output', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({'cand_records': cand_records, 'probs': probs}, f)
        print(f"Cached {len(probs):,} candidate pairs to {cache_path}")

    # Now evaluate strategies in 0.1s!
    strategies = [
        ("Baseline (0.87)", 0.87, False, 0.0),
        ("Baseline (0.88)", 0.88, False, 0.0),
        ("Baseline (0.89)", 0.89, False, 0.0),
        ("Baseline (0.90)", 0.90, False, 0.0),
        ("Baseline (0.91)", 0.91, False, 0.0),
        ("Baseline (0.92)", 0.92, False, 0.0),
        ("Thresh 0.87 + Filter Num Conflict", 0.87, True, 0.0),
        ("Thresh 0.88 + Filter Num Conflict", 0.88, True, 0.0),
        ("Thresh 0.89 + Filter Num Conflict", 0.89, True, 0.0),
        ("Thresh 0.90 + Filter Num Conflict", 0.90, True, 0.0),
        ("Thresh 0.91 + Filter Num Conflict", 0.91, True, 0.0),
        ("Thresh 0.92 + Filter Num Conflict", 0.92, True, 0.0),
        ("Thresh 0.87 + Filter Num + Min Name JW 0.60", 0.87, True, 0.60),
        ("Thresh 0.88 + Filter Num + Min Name JW 0.60", 0.88, True, 0.60),
        ("Thresh 0.89 + Filter Num + Min Name JW 0.60", 0.89, True, 0.60),
        ("Thresh 0.90 + Filter Num + Min Name JW 0.60", 0.90, True, 0.60),
        ("Thresh 0.91 + Filter Num + Min Name JW 0.60", 0.91, True, 0.60),
        ("Thresh 0.92 + Filter Num + Min Name JW 0.60", 0.92, True, 0.60),
        ("Thresh 0.88 + Filter Num + Min Name JW 0.65", 0.88, True, 0.65),
        ("Thresh 0.89 + Filter Num + Min Name JW 0.65", 0.89, True, 0.65),
        ("Thresh 0.90 + Filter Num + Min Name JW 0.65", 0.90, True, 0.65),
    ]

    print("\n" + "=" * 90)
    print(f"{'Strategy':<48} {'F0.5':<8} {'Prec':<8} {'Rec':<8} {'FP':<6} {'FN':<6} {'S_FP':<6}")
    print("=" * 90)

    best_f05 = 0.0
    best_strat = None

    for name, thresh, filter_num, min_jw in strategies:
        sid_cands = defaultdict(lambda: {'S2': [], 'S3': []})
        for i, (sid, cid, src, is_tp, s1_snum, csnum, s1_n, cn, s1_a, ca, score) in enumerate(cand_records):
            p = probs[i]
            if p < thresh: continue

            # Street number conflict check: both have numbers, but they differ
            if filter_num and s1_snum and csnum and s1_snum != csnum:
                continue

            # Minimum name JW check: prevent matching completely different businesses that share address/city
            if min_jw > 0.0:
                jw = JaroWinkler.similarity(s1_n, cn)
                if jw < min_jw:
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
        star = " *** BEST ***" if f05 > best_f05 else ""
        if f05 > best_f05:
            best_f05 = f05
            best_strat = name
        print(f"{name:<48} {f05:<8.4f} {p*100:<7.2f}% {r*100:<7.2f}% {fp:<6d} {fn:<6d} {s_fp:<6d}{star}")

    print("=" * 90)
    print(f"BEST STRATEGY: {best_strat} -> Macro F0.5 = {best_f05:.4f}")

if __name__ == '__main__':
    main()

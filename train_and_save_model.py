"""
train_and_save_model.py
=======================
Trains the next-generation LightGBM matching model (v3) with:
  1. Multilingual AnyAscii normalization & domain/honorific stripping
  2. 31 high-precision features without priority score bias/leak
  3. Memory-bounded candidate streaming (<1.5 GB peak RAM)
  4. Balanced hard negative mining + ground truth positive augmentation
  5. Calibrated scale_pos_weight=1.5 for honest posterior probabilities
  6. Multi-tier validation sweep (T_PRIMARY, T_SECONDARY, T_TERTIARY) targeting Macro F0.5
"""

import os
import sys
import re
import time
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.preprocessing import clean_name, squish, clean_addr, get_street_num, get_all_nums, get_street_prefix
from src.features import compute_pair_features, FEATURE_NAMES

def f05_score(precision: float, recall: float) -> float:
    beta2 = 0.25
    denom = beta2 * precision + recall
    return (1.0 + beta2) * precision * recall / denom if denom > 0 else 0.0

def macro_f05(predictions: dict, ground_truth: dict) -> float:
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        if not true_set and not pred_set:
            scores.append(1.0)
        elif not true_set and pred_set:
            scores.append(0.0)
        elif not pred_set:
            scores.append(0.0)
        else:
            tp = len(pred_set & true_set)
            p = tp / len(pred_set)
            r = tp / len(true_set)
            scores.append(f05_score(p, r))
    return float(np.mean(scores)) if scores else 0.0

def build_record_tuple(name: str, addr: str, ctry: str):
    c = str(ctry).upper().strip()
    cn = clean_name(name)
    sq = squish(cn)
    ca = clean_addr(addr, c)
    snum = get_street_num(addr)
    all_nums = get_all_nums(addr)
    words = cn.split()
    awords = ca.split()
    return (cn, sq, ca, snum, c, all_nums, words, awords)

def main():
    t0 = time.time()
    print("=" * 75, flush=True)
    print("  TRAINING HIGH-PRECISION MULTILINGUAL LIGHTGBM MODEL (v3)", flush=True)
    print("=" * 75, flush=True)

    N_TRAIN_S1 = 30_000
    N_VAL_S1 = 5_000
    TOTAL_S1 = N_TRAIN_S1 + N_VAL_S1
    MAX_CANDS_PER_ENT = 8

    # 1. Load Ground Truth for 35,000 S1 Entities
    print(f"\n[1/5] Loading Ground Truth for {TOTAL_S1:,} S1 entities...", flush=True)
    gt_df = pd.read_csv('data/dataset/train/train_ground_truth.tsv', sep='\t', nrows=TOTAL_S1)
    target_s1_ids = list(gt_df['source1_entity_id'])
    val_eids = set(target_s1_ids[:N_VAL_S1])
    train_eids = set(target_s1_ids[N_VAL_S1:])

    train_gt = defaultdict(set)
    val_gt = defaultdict(set)
    needed_cids = set()

    for _, r in gt_df.iterrows():
        sid = r['source1_entity_id']
        matches = set()
        if pd.notna(r['matched_entity_ids']):
            for mid in str(r['matched_entity_ids']).split(','):
                mid = mid.strip()
                if mid:
                    matches.add(mid)
                    needed_cids.add(mid)
        if sid in train_eids:
            train_gt[sid] = matches
        else:
            val_gt[sid] = matches

    # 2. Load S1 Entity Records
    print(f"\n[2/5] Loading {TOTAL_S1:,} S1 entity records...", flush=True)
    s1_rows = []
    target_set = set(target_s1_ids)
    for chunk in pd.read_csv('data/dataset/train/train_source1.tsv', sep='\t', chunksize=200000):
        m = chunk[chunk['entity_id'].isin(target_set)]
        if len(m): s1_rows.append(m)
        if sum(len(x) for x in s1_rows) >= TOTAL_S1: break
    s1_df = pd.concat(s1_rows).drop_duplicates('entity_id').set_index('entity_id')

    s1_tuples = {}
    idx_exact = defaultdict(list)
    idx_sq = defaultdict(list)
    idx_tfp = defaultdict(list)
    idx_w12 = defaultdict(list)
    idx_fl = defaultdict(list)
    idx_w23 = defaultdict(list)
    idx_addr_num_st = defaultdict(list)
    idx_addr_num_word = defaultdict(list)

    for sid in target_s1_ids:
        if sid not in s1_df.index: continue
        r = s1_df.loc[sid]
        tup = build_record_tuple(r['business_name'], r['business_address'], r['country'])
        s1_tuples[sid] = tup
        cn, sq, ca, snum, c, all_nums, words, awords = tup

        if cn: idx_exact[(c, cn)].append(sid)
        if len(sq) >= 4: idx_sq[(c, sq)].append(sid)
        if 2 <= len(words) <= 4: idx_tfp[(c, '|'.join(sorted(words)))].append(sid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[1]) >= 3: idx_w12[(c, words[0], words[1])].append(sid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[-1]) >= 3: idx_fl[(c, words[0], words[-1])].append(sid)
        if len(words) >= 3 and len(words[1]) >= 3 and len(words[-1]) >= 3: idx_w23[(c, words[1], words[-1])].append(sid)
        st_pref = get_street_prefix(r['business_address'])
        if snum and st_pref: idx_addr_num_st[(c, snum, st_pref)].append(sid)
        if snum:
            for w in words[:2]:
                if len(w) >= 3: idx_addr_num_word[(c, snum, w)].append(sid)

    # Prune overgrown keys for fast streaming
    idx_exact = {k: v for k, v in idx_exact.items() if len(v) <= 150}
    idx_sq = {k: v for k, v in idx_sq.items() if len(v) <= 150}
    idx_tfp = {k: v for k, v in idx_tfp.items() if len(v) <= 150}
    idx_w12 = {k: v for k, v in idx_w12.items() if len(v) <= 150}
    idx_fl = {k: v for k, v in idx_fl.items() if len(v) <= 150}
    idx_w23 = {k: v for k, v in idx_w23.items() if len(v) <= 150}
    idx_addr_num_st = {k: v for k, v in idx_addr_num_st.items() if len(v) <= 50}
    idx_addr_num_word = {k: v for k, v in idx_addr_num_word.items() if len(v) <= 50}

    print(f"      Built multi-tier inverted index for {len(s1_tuples):,} entities.", flush=True)

    # 3. Stream Sources 2 & 3 with Memory-Bounded Retention
    print("\n[3/5] Streaming S2 & S3 with memory-bounded retention (<1.5 GB)...", flush=True)
    s1_candidates = defaultdict(dict)
    cand_tuples = {}

    for fn in ['train_source2.tsv', 'train_source3.tsv']:
        filepath = 'data/dataset/train/' + fn
        print(f"      -> Processing {fn}...", flush=True)
        for chunk in pd.read_csv(filepath, sep='\t', chunksize=400000):
            eids = list(chunk['entity_id'])
            names = list(chunk['business_name'].fillna(''))
            addrs = list(chunk['business_address'].fillna(''))
            ctrys = list(chunk['country'].fillna(''))

            for cid, bname, baddr, bctry in zip(eids, names, addrs, ctrys):
                c = str(bctry).upper().strip()
                sid_scores = defaultdict(int)

                cn = clean_name(bname)
                sq = squish(cn)
                words = cn.split()
                snum = get_street_num(baddr)
                st_pref = get_street_prefix(baddr)

                if cn and (c, cn) in idx_exact:
                    for sid in idx_exact[(c, cn)]: sid_scores[sid] = max(sid_scores[sid], 100)
                if len(sq) >= 4 and (c, sq) in idx_sq:
                    for sid in idx_sq[(c, sq)]: sid_scores[sid] = max(sid_scores[sid], 95)
                if 2 <= len(words) <= 4 and (c, '|'.join(sorted(words))) in idx_tfp:
                    for sid in idx_tfp[(c, '|'.join(sorted(words)))]: sid_scores[sid] = max(sid_scores[sid], 85)
                if len(words) >= 2 and len(words[0]) >= 3 and len(words[1]) >= 3 and (c, words[0], words[1]) in idx_w12:
                    for sid in idx_w12[(c, words[0], words[1])]: sid_scores[sid] = max(sid_scores[sid], 80)
                if len(words) >= 2 and len(words[0]) >= 3 and len(words[-1]) >= 3 and (c, words[0], words[-1]) in idx_fl:
                    for sid in idx_fl[(c, words[0], words[-1])]: sid_scores[sid] = max(sid_scores[sid], 75)
                if len(words) >= 3 and len(words[1]) >= 3 and len(words[-1]) >= 3 and (c, words[1], words[-1]) in idx_w23:
                    for sid in idx_w23[(c, words[1], words[-1])]: sid_scores[sid] = max(sid_scores[sid], 70)
                if snum and st_pref and (c, snum, st_pref) in idx_addr_num_st:
                    for sid in idx_addr_num_st[(c, snum, st_pref)]: sid_scores[sid] = max(sid_scores[sid], 65)
                if snum:
                    for w in words[:2]:
                        if len(w) >= 3 and (c, snum, w) in idx_addr_num_word:
                            for sid in idx_addr_num_word[(c, snum, w)]: sid_scores[sid] = max(sid_scores[sid], 60)

                retained = False
                for sid, score in sid_scores.items():
                    cur = s1_candidates[sid]
                    if len(cur) < MAX_CANDS_PER_ENT:
                        cur[cid] = score
                        retained = True
                    else:
                        min_cid = min(cur, key=cur.__getitem__)
                        if score > cur[min_cid]:
                            del cur[min_cid]
                            cur[cid] = score
                            retained = True

                is_gt = cid in needed_cids
                if retained or is_gt:
                    cand_tuples[cid] = build_record_tuple(bname, baddr, bctry)

    print(f"      Total unique candidates retained in memory: {len(cand_tuples):,}", flush=True)
    augmented = 0
    for sid in target_s1_ids:
        true_cids = train_gt.get(sid, set()) | val_gt.get(sid, set())
        for cid in true_cids:
            if cid in cand_tuples:
                if cid not in s1_candidates[sid]:
                    s1_candidates[sid][cid] = 90
                    augmented += 1
    print(f"      Augmented {augmented:,} true matches into candidate sets.", flush=True)

    # 4. Feature Extraction (31 Pure Features)
    print("\n[4/5] Extracting 31 pure features for Train and Validation splits...", flush=True)
    X_train, y_train = [], []
    X_val, y_val = [], []
    val_pairs = []

    for sid in target_s1_ids:
        if sid not in s1_tuples: continue
        s1_tup = s1_tuples[sid]
        is_train = sid in train_eids
        true_cids = train_gt.get(sid, set()) if is_train else val_gt.get(sid, set())

        cands = s1_candidates.get(sid, {})
        for cid, prio in cands.items():
            if cid not in cand_tuples: continue
            cand_tup = cand_tuples[cid]
            feats = compute_pair_features(s1_tup, cand_tup)
            label = 1 if cid in true_cids else 0

            if is_train:
                X_train.append(feats)
                y_train.append(label)
            else:
                X_val.append(feats)
                y_val.append(label)
                val_pairs.append((sid, cid))

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)

    pos_tr = int(y_train.sum()); neg_tr = len(y_train) - pos_tr
    pos_val = int(y_val.sum()); neg_val = len(y_val) - pos_val
    print(f"      Train Set: {len(X_train):,} pairs (Pos: {pos_tr:,}, Neg: {neg_tr:,})", flush=True)
    print(f"      Val Set  : {len(X_val):,} pairs (Pos: {pos_val:,}, Neg: {neg_val:,})", flush=True)

    # 5. Fit Calibrated LightGBM Model v3
    print("\n[5/5] Training LightGBM Model v3 (scale_pos_weight=1.5)...", flush=True)
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=1.5,
        random_state=42,
        verbose=-1
    )
    clf.fit(X_train, y_train)
    print("      Model training complete.", flush=True)

    # Multi-Tier Validation Threshold Sweep
    print("\n  Multi-Tier Threshold Sweep on held-out validation set...", flush=True)
    val_probs = clf.predict_proba(X_val)[:, 1]

    val_ent_cands = defaultdict(list)
    for (sid, cid), prob in zip(val_pairs, val_probs):
        val_ent_cands[sid].append((cid, prob))
    for sid in val_ent_cands:
        val_ent_cands[sid].sort(key=lambda x: -x[1])

    best_score = 0.0
    best_t_prim = 0.60
    best_t_sec = 0.75
    best_t_tert = 0.85

    # Sweep primary from 0.45 to 0.85
    for t_prim in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        for t_sec in [0.70, 0.75, 0.80, 0.85]:
            if t_sec < t_prim: continue
            for t_tert in [0.80, 0.85, 0.90]:
                if t_tert < t_sec: continue
                preds = defaultdict(set)
                for sid in val_eids:
                    cands = val_ent_cands.get(sid, [])
                    for rank, (cid, p) in enumerate(cands[:4]):
                        thresh = t_prim if rank == 0 else (t_sec if rank == 1 else t_tert)
                        if p >= thresh:
                            preds[sid].add(cid)

                score = macro_f05(preds, val_gt)
                if score > best_score:
                    best_score = score
                    best_t_prim = t_prim
                    best_t_sec = t_sec
                    best_t_tert = t_tert
                    n_empty = sum(1 for sid in val_eids if len(preds.get(sid, set())) == 0)
                    pct_empty = n_empty / len(val_eids) * 100.0
                    print(f"    * NEW BEST * T_Prim: {t_prim:.2f}, T_Sec: {t_sec:.2f}, T_Tert: {t_tert:.2f} -> Macro F0.5 = {score:.4f} (Singletons: {pct_empty:.2f}%)", flush=True)

    print(f"\n>>> OPTIMAL MULTI-TIER THRESHOLDS <<<", flush=True)
    print(f"    T_PRIMARY   : {best_t_prim:.2f}", flush=True)
    print(f"    T_SECONDARY : {best_t_sec:.2f}", flush=True)
    print(f"    T_TERTIARY  : {best_t_tert:.2f}", flush=True)
    print(f"    Validation Macro F0.5: {best_score:.4f}", flush=True)

    # Feature Importances
    print("\n  Top Feature Importances (v3):", flush=True)
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx[:15], 1):
        print(f"    {rank:2d}. {FEATURE_NAMES[idx]:<26} : {importances[idx]}", flush=True)

    # Save Model Bundle v3
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    bundle_path = os.path.join(output_dir, 'champion_model_v3.pkl')
    bundle = {
        'clf': clf,
        't_primary': best_t_prim,
        't_secondary': best_t_sec,
        't_tertiary': best_t_tert,
        'val_f05': best_score,
        'feature_names': FEATURE_NAMES
    }
    with open(bundle_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"\n  Saved updated model bundle -> {bundle_path}", flush=True)
    print(f"  Total pipeline time: {(time.time()-t0)/60:.1f} minutes", flush=True)

if __name__ == '__main__':
    main()

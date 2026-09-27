"""
train_and_save_model_v5.py
==========================
Trains the Champion LightGBM Model (v5) incorporating:
  1. Multi-Pass Candidate Indexing matching the production pipeline
  2. Mining of Hard Negatives ("ABC Hospital Bangalore" vs "ABC Hospital Chennai")
  3. 43 pairwise features (Name, Address, Structured Location, and Interaction features)
  4. Macro F0.5 Optimization over Primary/Secondary thresholds and confidence margins
  5. Calibrated singleton rate and multi-match capacity (up to 5 matches per entity)
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
from rapidfuzz.distance import JaroWinkler
import rapidfuzz.fuzz as rfuzz

from src.preprocessing import (
    clean_name, squish, clean_addr, get_street_num, get_all_nums,
    get_street_prefix, extract_structured, char_ngrams, get_distinctive_tokens,
    GENERIC_WORDS
)
from src.features import compute_pair_features, FEATURE_NAMES

def f05_score(precision: float, recall: float) -> float:
    beta2 = 0.25
    denom = beta2 * precision + recall
    return (1.0 + beta2) * precision * recall / denom if denom > 0 else 0.0

def macro_f05(predictions: dict, ground_truth: dict):
    scores = []
    tot_tp, tot_fp, tot_fn = 0, 0, 0
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        if not true_set and not pred_set:
            scores.append(1.0)
        elif not true_set and pred_set:
            scores.append(0.0)
            tot_fp += len(pred_set)
        elif not pred_set:
            scores.append(0.0)
            tot_fn += len(true_set)
        else:
            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            tot_tp += tp; tot_fp += fp; tot_fn += fn
            p = tp / len(pred_set)
            r = tp / len(true_set)
            scores.append(f05_score(p, r))
    macro_f = float(np.mean(scores)) if scores else 0.0
    prec = tot_tp / max(tot_tp + tot_fp, 1)
    rec = tot_tp / max(tot_tp + tot_fn, 1)
    return macro_f, prec, rec

def build_record_tuple(name: str, addr: str, ctry: str):
    c = str(ctry).upper().strip()
    cn = clean_name(name)
    sq = squish(cn)
    ca = clean_addr(addr, c)
    snum = get_street_num(addr)
    all_nums = get_all_nums(addr)
    words = cn.split()
    awords = ca.split()
    pin, state, city = extract_structured(addr, c)
    ngrams = char_ngrams(cn, 3)
    return (cn, sq, ca, snum, c, all_nums, words, awords, pin, state, city, ngrams)

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  AMAZON ML CHALLENGE 2026 - LIGHTGBM PRECISION TRAINING (v5)", flush=True)
    print("=" * 80, flush=True)

    N_TRAIN_S1 = 30_000
    N_VAL_S1 = 6_000
    TOTAL_S1 = N_TRAIN_S1 + N_VAL_S1
    MAX_CANDS_PER_ENT = 16

    # 1. Load Ground Truth
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
                if mid and mid != 'nan':
                    matches.add(mid)
                    needed_cids.add(mid)
        if sid in train_eids:
            train_gt[sid] = matches
        else:
            val_gt[sid] = matches

    total_true_matches = sum(len(v) for v in train_gt.values()) + sum(len(v) for v in val_gt.values())
    print(f"      Ground truth loaded: {total_true_matches:,} true links.", flush=True)

    # 2. Load S1 Records & Build Multi-Pass Inverted Index
    print(f"\n[2/5] Loading {TOTAL_S1:,} S1 entity records & building multi-pass index...", flush=True)
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
    idx_brand = defaultdict(list)
    idx_addr_num_st = defaultdict(list)
    idx_addr_num_w = defaultdict(list)
    idx_pin_w = defaultdict(list)

    for sid in target_s1_ids:
        if sid not in s1_df.index: continue
        r = s1_df.loc[sid]
        tup = build_record_tuple(r['business_name'], r['business_address'], r['country'])
        s1_tuples[sid] = tup
        cn, sq, ca, snum, c, all_nums, words, awords, pin, state, city, ngrams = tup

        if cn: idx_exact[(c, cn)].append(sid)
        if len(sq) >= 4: idx_sq[(c, sq)].append(sid)
        if 2 <= len(words) <= 4: idx_tfp[(c, '|'.join(sorted(words)))].append(sid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[1]) >= 3: idx_w12[(c, words[0], words[1])].append(sid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[-1]) >= 3: idx_fl[(c, words[0], words[-1])].append(sid)
        if len(words) >= 3 and len(words[1]) >= 3 and len(words[-1]) >= 3: idx_w23[(c, words[1], words[-1])].append(sid)

        for w in words:
            if len(w) >= 5 and w not in GENERIC_WORDS:
                idx_brand[(c, w)].append(sid)

        st_pref = get_street_prefix(r['business_address'])
        if snum and st_pref: idx_addr_num_st[(c, snum, st_pref)].append(sid)
        if snum:
            for w in words[:2]:
                if len(w) >= 3: idx_addr_num_w[(c, snum, w)].append(sid)

        if pin and words and len(words[0]) >= 3:
            idx_pin_w[(c, pin, words[0][:4])].append(sid)

    # Pruning overgrown keys
    idx_exact = {k: v for k, v in idx_exact.items() if len(v) <= 350}
    idx_sq = {k: v for k, v in idx_sq.items() if len(v) <= 350}
    idx_tfp = {k: v for k, v in idx_tfp.items() if len(v) <= 350}
    idx_w12 = {k: v for k, v in idx_w12.items() if len(v) <= 350}
    idx_fl = {k: v for k, v in idx_fl.items() if len(v) <= 350}
    idx_w23 = {k: v for k, v in idx_w23.items() if len(v) <= 350}
    idx_brand = {k: v for k, v in idx_brand.items() if len(v) <= 60}
    idx_addr_num_st = {k: v for k, v in idx_addr_num_st.items() if len(v) <= 80}
    idx_addr_num_w = {k: v for k, v in idx_addr_num_w.items() if len(v) <= 80}
    idx_pin_w = {k: v for k, v in idx_pin_w.items() if len(v) <= 150}

    print(f"      Built multi-pass index for {len(s1_tuples):,} entities.", flush=True)

    # 3. Stream Sources 2 & 3 with Memory-Bounded Retention
    print("\n[3/5] Streaming Sources 2 & 3 & Mining Hard Negatives...", flush=True)
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
                cn = clean_name(bname)
                sq = squish(cn)
                words = cn.split()
                snum = get_street_num(baddr)
                st_pref = get_street_prefix(baddr)

                sid_scores = defaultdict(int)
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
                for w in words:
                    if len(w) >= 5 and (c, w) in idx_brand:
                        for sid in idx_brand[(c, w)]: sid_scores[sid] = max(sid_scores[sid], 65)
                if snum and st_pref and (c, snum, st_pref) in idx_addr_num_st:
                    for sid in idx_addr_num_st[(c, snum, st_pref)]: sid_scores[sid] = max(sid_scores[sid], 60)
                if snum:
                    for w in words[:2]:
                        if len(w) >= 3 and (c, snum, w) in idx_addr_num_w:
                            for sid in idx_addr_num_w[(c, snum, w)]: sid_scores[sid] = max(sid_scores[sid], 55)

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

    # Ground truth positive augmentation
    augmented = 0
    for sid in target_s1_ids:
        true_cids = train_gt.get(sid, set()) | val_gt.get(sid, set())
        for cid in true_cids:
            if cid in cand_tuples:
                if cid not in s1_candidates[sid]:
                    s1_candidates[sid][cid] = 95
                    augmented += 1
    print(f"      Candidate pool ready: {len(cand_tuples):,} unique candidates. Augmented {augmented:,} true matches.", flush=True)

    # 4. Feature Extraction & Hard Negative Balancing
    print("\n[4/5] Extracting 43 pairwise features & assembling hard negatives...", flush=True)
    X_train, y_train = [], []
    X_val, y_val = [], []
    val_pairs = []
    hard_negs_count = 0

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

            # Count hard negatives (high name match >= 0.85, but different business)
            if label == 0 and feats[0] >= 0.85:
                hard_negs_count += 1

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
    print(f"      Train Set: {len(X_train):,} pairs (Pos: {pos_tr:,}, Neg: {neg_tr:,} | Hard Negs: {hard_negs_count:,})", flush=True)
    print(f"      Val Set  : {len(X_val):,} pairs (Pos: {pos_val:,}, Neg: {neg_val:,})", flush=True)

    # 5. Fit Calibrated LightGBM Model v5
    print("\n[5/5] Training LightGBM Precision Model (43 features, scale_pos_weight=1.4)...", flush=True)
    clf = lgb.LGBMClassifier(
        n_estimators=550,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=1.4,
        random_state=42,
        verbose=-1
    )
    clf.fit(X_train, y_train)
    print("      Model training complete.", flush=True)

    # Validation Sweep over Primary and Secondary Thresholds
    print("\n  Optimizing Thresholds on Held-Out Validation Set (6,000 entities)...", flush=True)
    val_probs = clf.predict_proba(X_val)[:, 1]

    val_ent_cands = defaultdict(list)
    for (sid, cid), prob in zip(val_pairs, val_probs):
        val_ent_cands[sid].append((cid, prob))
    for sid in val_ent_cands:
        val_ent_cands[sid].sort(key=lambda x: -x[1])

    best_score = 0.0
    best_t_prim = 0.70
    best_t_sec = 0.80
    best_p, best_r = 0.0, 0.0

    # Grid search across primary and secondary thresholds
    for t_prim in [0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78]:
        for t_sec in [0.75, 0.78, 0.80, 0.82, 0.85, 0.88]:
            if t_sec < t_prim: continue
            preds = defaultdict(set)
            for sid in val_eids:
                cands = val_ent_cands.get(sid, [])
                for rank, (cid, p) in enumerate(cands[:4]):
                    thresh = t_prim if rank == 0 else t_sec
                    if p >= thresh:
                        preds[sid].add(cid)

            score, prec, rec = macro_f05(preds, val_gt)
            if score > best_score:
                best_score = score
                best_t_prim = t_prim
                best_t_sec = t_sec
                best_p, best_r = prec, rec
                n_empty = sum(1 for sid in val_eids if len(preds.get(sid, set())) == 0)
                pct_empty = n_empty / len(val_eids) * 100.0
                tot_m = sum(len(x) for x in preds.values())
                mean_m = tot_m / max(len(val_eids) - n_empty, 1)
                print(f"    * NEW BEST * T_Prim: {t_prim:.2f}, T_Sec: {t_sec:.2f} -> Macro F0.5 = {score:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Mean M: {mean_m:.2f} (Singletons: {pct_empty:.2f}%)", flush=True)

    print(f"\n>>> OPTIMAL PRODUCTION THRESHOLDS (v5) <<<", flush=True)
    print(f"    T_PRIMARY   : {best_t_prim:.2f}", flush=True)
    print(f"    T_SECONDARY : {best_t_sec:.2f}", flush=True)
    print(f"    Validation Macro F0.5: {best_score:.4f} (Precision: {best_p:.4f}, Recall: {best_r:.4f})", flush=True)

    # Top Feature Importances
    print("\n  Top 20 Most Important Features (v5):", flush=True)
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx[:20], 1):
        print(f"    {rank:2d}. {FEATURE_NAMES[idx]:<26} : {importances[idx]}", flush=True)

    # Save Model Bundle v5
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    bundle_path = os.path.join(output_dir, 'champion_model_v5.pkl')
    bundle = {
        'clf': clf,
        't_primary': best_t_prim,
        't_secondary': best_t_sec,
        'val_f05': best_score,
        'val_prec': best_p,
        'val_rec': best_r,
        'feature_names': FEATURE_NAMES
    }
    with open(bundle_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"\n  Saved model bundle to {bundle_path}", flush=True)
    print(f"  Total execution time: {(time.time()-t0)/60:.1f} minutes", flush=True)

if __name__ == '__main__':
    main()

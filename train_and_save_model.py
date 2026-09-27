import os
import re
import time
import pickle
import unicodedata
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz.distance import JaroWinkler
import rapidfuzz.fuzz as rfuzz

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

def f05_score(precision: float, recall: float) -> float:
    beta2 = 0.25
    denom = beta2 * precision + recall
    return (1 + beta2) * precision * recall / denom if denom > 0 else 0.0

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

FEATURE_COLS = ['jw_n', 'ts_n', 'tset_n', 'pr_n', 'exact_n', 'exact_sq', 'num_m', 'jw_a', 'tset_a', 'country_m', 'prio_score']

def main():
    print("=" * 70)
    print("  TRAINING HIGH-PRECISION LIGHTGBM MATCHING MODEL")
    print("=" * 70)
    t0 = time.time()
    
    # 1. Load 35,000 S1 entities and ground truth (30k train, 5k val)
    gt_df = pd.read_csv('data/dataset/train/train_ground_truth.tsv', sep='\t', nrows=35000)
    target_s1_ids = set(gt_df['source1_entity_id'])

    s1_rows = []
    for chunk in pd.read_csv('data/dataset/train/train_source1.tsv', sep='\t', chunksize=200000):
        m = chunk[chunk['entity_id'].isin(target_s1_ids)]
        if len(m): s1_rows.append(m)
        if sum(len(x) for x in s1_rows) >= len(target_s1_ids): break
    s1_df = pd.concat(s1_rows).drop_duplicates('entity_id')

    all_eids = list(s1_df['entity_id'].unique())
    np.random.seed(42)
    np.random.shuffle(all_eids)
    val_eids = set(all_eids[:5000])
    train_eids = set(all_eids[5000:])

    train_s1 = s1_df[s1_df['entity_id'].isin(train_eids)].set_index('entity_id')
    val_s1   = s1_df[s1_df['entity_id'].isin(val_eids)].set_index('entity_id')

    train_gt = defaultdict(set)
    val_gt   = defaultdict(set)
    for _, r in gt_df.iterrows():
        sid = r['source1_entity_id']
        m = set(x.strip() for x in str(r['matched_entity_ids']).split(',') if x.strip()) if pd.notna(r['matched_entity_ids']) else set()
        if sid in train_eids: train_gt[sid] = m
        elif sid in val_eids: val_gt[sid] = m

    print(f"Loaded {len(train_s1):,} Train S1 entities and {len(val_s1):,} Val S1 entities in {time.time()-t0:.1f}s")

    # 2. Build multi-key index for S1 entities
    idx_name = defaultdict(list); idx_sq = defaultdict(list); idx_fl = defaultdict(list)
    idx_w12 = defaultdict(list); idx_w23 = defaultdict(list); idx_tfp = defaultdict(list)
    idx_addr_num_st = defaultdict(list); idx_addr_num_name = defaultdict(list)

    for sid, r in s1_df.set_index('entity_id').iterrows():
        c = str(r['country']).upper().strip()
        cn = clean_name(r['business_name']); sq = squish(cn)
        snum = get_street_num(r['business_address'])
        st_pref = get_street_prefix(r['business_address'])
        words = cn.split()
        if cn: idx_name[(c, cn)].append(sid)
        if len(sq) >= 5: idx_sq[(c, sq)].append(sid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[-1]) >= 3: idx_fl[(c, words[0], words[-1])].append(sid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[1]) >= 3: idx_w12[(c, words[0], words[1])].append(sid)
        if len(words) >= 3 and len(words[1]) >= 3 and len(words[-1]) >= 3: idx_w23[(c, words[1], words[-1])].append(sid)
        if 2 <= len(words) <= 4: idx_tfp[(c, '|'.join(sorted(words)))].append(sid)
        if snum and st_pref: idx_addr_num_st[(c, snum, st_pref)].append(sid)
        if snum and len(words) >= 1 and len(words[0]) >= 3: idx_addr_num_name[(c, snum, words[0])].append(sid)

    # 3. Stream S2 & S3 and retain prioritized candidates
    s2_cands = defaultdict(dict)
    s3_cands = defaultdict(dict)
    t_stream = time.time()

    for fn in ['train_source2.tsv', 'train_source3.tsv']:
        target_dict = s2_cands if 'source2' in fn else s3_cands
        for chunk in pd.read_csv('data/dataset/train/' + fn, sep='\t', chunksize=300000):
            for cid, bname, baddr, bctry in zip(chunk['entity_id'], chunk['business_name'], chunk['business_address'], chunk['country']):
                c = str(bctry).upper().strip()
                cn = clean_name(bname); sq = squish(cn)
                snum = get_street_num(baddr)
                words = cn.split()
                st_pref = get_street_prefix(baddr)
                
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
                    
                for sid, score in sid_scores.items():
                    cur = target_dict[sid]
                    if len(cur) < 8:
                        cur[cid] = (score, bname, baddr, bctry)
                    else:
                        min_cid = min(cur.keys(), key=lambda x: cur[x][0])
                        if score > cur[min_cid][0]:
                            del cur[min_cid]
                            cur[cid] = (score, bname, baddr, bctry)

    print(f"Streaming completed in {time.time()-t_stream:.1f}s")

    # 4. Feature Extraction & Training
    train_X, train_y = [], []
    for sid in train_eids:
        s1_r = train_s1.loc[sid]
        s1_n = clean_name(s1_r['business_name']); s1_sq = squish(s1_n)
        s1_a = clean_addr(s1_r['business_address']); s1_snum = get_street_num(s1_r['business_address'])
        s1_c = str(s1_r['country']).upper().strip()
        true_cids = train_gt.get(sid, set())
        
        cands = {**s2_cands.get(sid, {}), **s3_cands.get(sid, {})}
        for cid, (score, bname, baddr, bctry) in cands.items():
            cn = clean_name(bname); csq = squish(cn)
            ca = clean_addr(baddr); csnum = get_street_num(baddr)
            cc = str(bctry).upper().strip()
            feats = compute_features(s1_n, s1_sq, s1_a, s1_snum, s1_c, cn, csq, ca, csnum, cc, score)
            train_X.append(feats)
            train_y.append(1 if cid in true_cids else 0)

    train_X = np.array(train_X, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.int32)
    pos = int(train_y.sum())
    neg = len(train_y) - pos
    print(f"Training set: {len(train_X):,} pairs (Pos: {pos:,}, Neg: {neg:,})")

    scale = neg / max(pos, 1)
    clf = lgb.LGBMClassifier(n_estimators=450, learning_rate=0.04, num_leaves=63, scale_pos_weight=scale, random_state=42, verbose=-1)
    clf.fit(train_X, train_y)
    print("LightGBM fit complete.")

    # 5. Threshold Tuning on Validation
    print("\nTuning threshold on held-out validation set...")
    val_meta, val_rows = [], []
    for sid in val_eids:
        s1_r = val_s1.loc[sid]
        s1_n = clean_name(s1_r['business_name']); s1_sq = squish(s1_n)
        s1_a = clean_addr(s1_r['business_address']); s1_snum = get_street_num(s1_r['business_address'])
        s1_c = str(s1_r['country']).upper().strip()
        
        cands = {**s2_cands.get(sid, {}), **s3_cands.get(sid, {})}
        for cid, (score, bname, baddr, bctry) in cands.items():
            cn = clean_name(bname); csq = squish(cn)
            ca = clean_addr(baddr); csnum = get_street_num(baddr)
            cc = str(bctry).upper().strip()
            val_rows.append(compute_features(s1_n, s1_sq, s1_a, s1_snum, s1_c, cn, csq, ca, csnum, cc, score))
            val_meta.append((sid, cid))

    val_X = np.array(val_rows, dtype=np.float32)
    probs = clf.predict_proba(val_X)[:, 1]

    best_t, best_f05 = 0.5, 0.0
    for thresh in np.arange(0.50, 0.95, 0.02):
        preds = defaultdict(set)
        for (sid, cid), p in zip(val_meta, probs):
            if p >= thresh:
                preds[sid].add(cid)
        f05 = macro_f05(preds, val_gt)
        if f05 > best_f05:
            best_f05 = f05
            best_t = thresh

    print(f"Optimal Threshold: {best_t:.2f} (Macro F0.5: {best_f05:.4f})")

    # Save trained model bundle
    os.makedirs('output', exist_ok=True)
    bundle = {
        'clf': clf,
        'threshold': best_t,
        'val_f05': best_f05,
        'feature_cols': FEATURE_COLS
    }
    model_path = 'output/champion_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"Model saved -> {model_path} in {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()

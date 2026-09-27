import os
import sys
import json
import time
import pickle
import unicodedata
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
import rapidfuzz.fuzz as rfuzz

sys.path.append('.')
from src.eval_utils import evaluate_entity_macro_f05
from src.data_split import get_or_create_val_split

# v2: city token helper for new address-based tiers
def get_city_tokens(addr: str) -> list:
    """Extract city/area tokens from address tail (addresses end with city/state)."""
    if not isinstance(addr, str) or not addr: return []
    s = unicodedata.normalize('NFKD', addr).encode('ASCII', 'ignore').decode('utf-8').lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\b(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|way|suite|ste|apt|floor|fl)\b', ' ', s)
    words = [w for w in s.split() if not w.isdigit() and len(w) >= 4]
    # Use LAST 3 tokens — most addresses end with locality/city/state
    return words[-3:] if len(words) >= 3 else words

def clean_name(s: str) -> str:
    if not isinstance(s, str) or not s: return ''
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8').lower()
    for sep in [' d/b/a ', ' dba ', ' t/a ', ' ta ', ' trading as ']:
        if sep in s: s = s.split(sep)[-1]; break
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\b(inc|corp|corporation|incorporated|llc|pllc|ltd|limited|co|company|pvt|private|llp|pc|sarl|sas|sasu|sa|eurl|snc|sci|gie)\b', ' ', s)
    return ' '.join(s.split())

def squish(s: str) -> str:
    s = re.sub(r'\b(com|org|net|in|fr|io|co|biz|info)\b', '', s)
    return s.replace(' ', '')

def clean_addr(s: str) -> str:
    if not isinstance(s, str) or not s: return ''
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8').lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\b(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|way|suite|ste|apt|floor|fl)\b', ' ', s)
    return ' '.join(s.split())

def get_street_num(addr: str) -> str:
    if not isinstance(addr, str) or not addr: return ''
    nums = re.findall(r'\b\d+\b', addr)
    return nums[0] if nums else ''

def get_street_prefix(addr: str) -> str:
    if not isinstance(addr, str) or not addr: return ''
    ca = clean_name(addr)
    words = [w for w in ca.split() if not w.isdigit() and len(w) >= 3]
    return words[0][:3] if words else ''

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

def main():
    print("=" * 70)
    print("  PHASE 1 v3: VERIFY EVALUATION & METRICS (10-Tier, No Exact Shortcut)")
    print("=" * 70)
    t0 = time.time()
    
    # 1. Load validation IDs and ground truth
    val_s1_ids = get_or_create_val_split(n_val=10000)
    val_set = set(val_s1_ids)
    
    gt_df = pd.read_csv('data/dataset/train/train_ground_truth.tsv', sep='\t', nrows=10000)
    ground_truth = {}
    total_true_matches = 0
    all_true_cids = set()
    
    for _, r in gt_df.iterrows():
        sid = r['source1_entity_id']
        matches = set()
        if pd.notna(r['matched_entity_ids']):
            for mid in str(r['matched_entity_ids']).split(','):
                mid = mid.strip()
                if mid:
                    matches.add(mid)
                    all_true_cids.add(mid)
        ground_truth[sid] = matches
        total_true_matches += len(matches)
        
    print(f"Validation Ground Truth: {len(val_s1_ids):,} S1 entities ({total_true_matches:,} true pairs, {len(all_true_cids):,} unique targets)")
    
    # 2. Load trained model
    with open('output/champion_model.pkl', 'rb') as f:
        bundle = pickle.load(f)
    clf = bundle['clf']
    threshold = bundle.get('threshold', 0.82)
    print(f"Loaded LightGBM model. Calibrated threshold: {threshold:.2f}")

    # 3. Load S1 data
    s1_rows = []
    for chunk in pd.read_csv('data/dataset/train/train_source1.tsv', sep='\t', chunksize=200000):
        m = chunk[chunk['entity_id'].isin(val_set)]
        if len(m): s1_rows.append(m)
        if sum(len(x) for x in s1_rows) >= len(val_set): break
    s1_df = pd.concat(s1_rows).drop_duplicates('entity_id').set_index('entity_id')

    # Build S1 query index (v2: 10-tier)
    idx_name = defaultdict(list); idx_sq = defaultdict(list); idx_fl = defaultdict(list)
    idx_w12 = defaultdict(list); idx_w23 = defaultdict(list); idx_tfp = defaultdict(list)
    idx_addr_num_st = defaultdict(list); idx_addr_num_name = defaultdict(list)
    idx_city_w12 = defaultdict(list)   # NEW Tier 9
    idx_num_city_w1 = defaultdict(list) # NEW Tier 10
    s1_clean_dict = {}

    for sid, r in s1_df.iterrows():
        c = str(r['country']).upper().strip()
        cn = clean_name(r['business_name']); sq = squish(cn)
        ca = clean_addr(r['business_address'])
        snum = get_street_num(r['business_address'])
        st_pref = get_street_prefix(r['business_address'])
        city_tok = get_city_tokens(r['business_address'])
        words = cn.split()
        s1_clean_dict[sid] = (cn, sq, ca, snum, c)

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

    # 4. Stream S2 & S3 and retain candidates (v2: 10-tier, cap=10 per source)
    s2_cands = defaultdict(dict)
    s3_cands = defaultdict(dict)
    all_retained_candidates = defaultdict(set)
    total_candidates_generated = 0
    MAX_CANDS_PER_SRC = 25  # v3: raised to 25/source for better recall coverage

    print("Streaming candidate records from S2 and S3 (v3: 10-tier, cap=25)...")
    for fn in ['train_source2.tsv', 'train_source3.tsv']:
        is_s2 = 'source2' in fn
        target_dict = s2_cands if is_s2 else s3_cands
        for chunk in pd.read_csv('data/dataset/train/' + fn, sep='\t', chunksize=400000):
            for cid, bname, baddr, bctry in zip(chunk['entity_id'], chunk['business_name'], chunk['business_address'], chunk['country']):
                c = str(bctry).upper().strip()
                cn = clean_name(bname); sq = squish(cn)
                ca = clean_addr(baddr); snum = get_street_num(baddr)
                words = cn.split()
                st_pref = get_street_prefix(baddr)
                city_tok = get_city_tokens(baddr)

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
                # NEW Tier 9: city word pair
                if len(city_tok) >= 2 and (c, city_tok[0], city_tok[1]) in idx_city_w12:
                    for sid in idx_city_w12[(c, city_tok[0], city_tok[1])]: sid_scores[sid] = max(sid_scores[sid], 55)
                # NEW Tier 10: house_num + city_w1
                if snum and len(city_tok) >= 1 and (c, snum, city_tok[0]) in idx_num_city_w1:
                    for sid in idx_num_city_w1[(c, snum, city_tok[0])]: sid_scores[sid] = max(sid_scores[sid], 50)

                for sid, score in sid_scores.items():
                    cur = target_dict[sid]
                    if len(cur) < MAX_CANDS_PER_SRC:
                        cur[cid] = (score, cn, sq, ca, snum, c)
                    else:
                        min_cid = min(cur.keys(), key=lambda x: cur[x][0])
                        if score > cur[min_cid][0]:
                            del cur[min_cid]
                            cur[cid] = (score, cn, sq, ca, snum, c)

    # 5. Measure Blocking Metrics
    candidate_true_positives = 0
    entities_with_hit = 0
    for sid in val_s1_ids:
        cands = set(s2_cands.get(sid, {}).keys()) | set(s3_cands.get(sid, {}).keys())
        all_retained_candidates[sid] = cands
        total_candidates_generated += len(cands)
        true_set = ground_truth.get(sid, set())
        tp = len(cands & true_set)
        candidate_true_positives += tp
        if tp > 0 or (len(true_set) == 0):
            entities_with_hit += 1

    blocking_pair_recall = candidate_true_positives / total_true_matches if total_true_matches else 0.0
    entity_coverage = entities_with_hit / len(val_s1_ids)
    candidate_precision = candidate_true_positives / total_candidates_generated if total_candidates_generated else 0.0

    # 6. Current Pipeline Prediction Logic (Exact shortcut + Model threshold 0.82 + source cap 3)
    # v3: NO exact match shortcut — all pairs go through LightGBM to prevent FP explosion
    batch_features = []
    batch_pairs = []

    for sid in val_s1_ids:
        if sid not in s1_clean_dict: continue
        s1_n, s1_sq, s1_a, s1_snum, s1_c = s1_clean_dict[sid]
        s2_dict = s2_cands.get(sid, {})
        s3_dict = s3_cands.get(sid, {})

        for cid, (score, cn, csq, ca, csnum, cc) in s2_dict.items():
            batch_features.append(compute_features(s1_n, s1_sq, s1_a, s1_snum, s1_c, cn, csq, ca, csnum, cc, score))
            batch_pairs.append((sid, cid, 'S2'))

        for cid, (score, cn, csq, ca, csnum, cc) in s3_dict.items():
            batch_features.append(compute_features(s1_n, s1_sq, s1_a, s1_snum, s1_c, cn, csq, ca, csnum, cc, score))
            batch_pairs.append((sid, cid, 'S3'))

    s1_model_matches = defaultdict(lambda: {'S2': list(), 'S3': list()})
    if batch_features:
        X_fuzzy = np.array(batch_features, dtype=np.float32)
        fuzzy_probs = clf.predict_proba(X_fuzzy)[:, 1]
        for (sid, cid, src), prob in zip(batch_pairs, fuzzy_probs):
            if prob >= threshold:
                s1_model_matches[sid][src].append((cid, prob))

    predictions = {}
    for sid in val_s1_ids:
        s2_model = [cid for cid, _ in sorted(s1_model_matches[sid]['S2'], key=lambda x: -x[1])]
        s2_final = s2_model[:3]
        s3_model = [cid for cid, _ in sorted(s1_model_matches[sid]['S3'], key=lambda x: -x[1])]
        s3_final = s3_model[:3]
        predictions[sid] = set(s2_final + s3_final)

    # 7. Evaluate exact macro F0.5
    eval_res = evaluate_entity_macro_f05(predictions, ground_truth, val_s1_ids)

    # 8. Assemble validation_metrics.json
    val_metrics = {
        'blocking_pair_recall': blocking_pair_recall,
        'entity_coverage': entity_coverage,
        'candidate_count': total_candidates_generated,
        'candidate_precision': candidate_precision,
        'pair_precision': eval_res['pair_precision'],
        'pair_recall': eval_res['pair_recall'],
        'macro_f05': eval_res['macro_f05'],
        'number_of_singletons': eval_res['number_of_singletons'],
        'number_of_predicted_singletons': eval_res['number_of_predicted_singletons'],
        'false_positive_count': eval_res['false_positive_count'],
        'false_negative_count': eval_res['false_negative_count'],
        'false_match_on_singletons': eval_res['false_match_on_singletons'],
        'correctly_rejected_singletons': eval_res['correctly_rejected_singletons'],
    }

    with open('validation_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(val_metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("          PHASE 1: VALIDATION METRICS JSON GENERATED")
    print("=" * 70)
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<32}: {v:.4f} ({v*100:.2f}%)")
        else:
            print(f"  {k:<32}: {v:,}")
    print("=" * 70)
    print(f"Metrics saved -> validation_metrics.json in {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()

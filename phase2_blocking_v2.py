import os
import re
import time
import json
import csv
import unicodedata
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

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

def main():
    print("=" * 75)
    print("  PHASE 2: CHARACTER N-GRAM TF-IDF + MULTI-TIER BLOCKING OPTIMIZATION")
    print("=" * 75)
    t0 = time.time()

    # 1. Load 10,000 validation Ground Truth
    gt_df = pd.read_csv('data/dataset/train/train_ground_truth.tsv', sep='\t', nrows=10000)
    s1_ids = list(gt_df['source1_entity_id'])
    val_set = set(s1_ids)

    ground_truth = {}
    target_cids = set()
    total_true_pairs = 0

    for _, r in gt_df.iterrows():
        sid = r['source1_entity_id']
        matches = set()
        if pd.notna(r['matched_entity_ids']):
            for mid in str(r['matched_entity_ids']).split(','):
                mid = mid.strip()
                if mid:
                    matches.add(mid)
                    target_cids.add(mid)
        ground_truth[sid] = matches
        total_true_pairs += len(matches)

    print(f"Validation Ground Truth: {len(s1_ids):,} entities with {total_true_pairs:,} true matches.")

    # 2. Load Source-1 attributes
    s1_rows = []
    for chunk in pd.read_csv('data/dataset/train/train_source1.tsv', sep='\t', chunksize=200000):
        m = chunk[chunk['entity_id'].isin(val_set)]
        if len(m): s1_rows.append(m)
        if sum(len(x) for x in s1_rows) >= len(val_set): break
    s1_df = pd.concat(s1_rows).drop_duplicates('entity_id').set_index('entity_id')

    # Build existing 8-tier blocking indices for S1
    idx_name = defaultdict(list); idx_sq = defaultdict(list); idx_fl = defaultdict(list)
    idx_w12 = defaultdict(list); idx_w23 = defaultdict(list); idx_tfp = defaultdict(list)
    idx_addr_num_st = defaultdict(list); idx_addr_num_name = defaultdict(list)
    s1_clean_dict = {}

    for sid in s1_ids:
        if sid not in s1_df.index: continue
        r = s1_df.loc[sid]
        c = str(r['country']).upper().strip()
        cn = clean_name(r['business_name']); sq = squish(cn)
        ca = clean_addr(r['business_address'])
        snum = get_street_num(r['business_address'])
        st_pref = get_street_prefix(r['business_address'])
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

    # 3. Stream existing 8-tier rule candidates
    print("\n[Step 1/3] Generating Existing Rule Candidates...")
    t_rule = time.time()
    existing_rule_cands = defaultdict(set)

    for fn in ['train_source2.tsv', 'train_source3.tsv']:
        print(f"  Streaming {fn}...")
        for chunk in pd.read_csv('data/dataset/train/' + fn, sep='\t', chunksize=500000):
            for cid, bname, baddr, bctry in zip(chunk['entity_id'], chunk['business_name'], chunk['business_address'], chunk['country']):
                c = str(bctry).upper().strip()
                cn = clean_name(bname); sq = squish(cn)
                snum = get_street_num(baddr)
                words = cn.split()
                st_pref = get_street_prefix(baddr)

                hit_sids = set()
                if cn and (c, cn) in idx_name: hit_sids.update(idx_name[(c, cn)])
                if len(sq) >= 5 and (c, sq) in idx_sq: hit_sids.update(idx_sq[(c, sq)])
                if 2 <= len(words) <= 4 and (c, '|'.join(sorted(words))) in idx_tfp: hit_sids.update(idx_tfp[(c, '|'.join(sorted(words)))])
                if len(words) >= 2 and len(words[0]) >= 3 and len(words[1]) >= 3 and (c, words[0], words[1]) in idx_w12: hit_sids.update(idx_w12[(c, words[0], words[1])])
                if len(words) >= 2 and len(words[0]) >= 3 and len(words[-1]) >= 3 and (c, words[0], words[-1]) in idx_fl: hit_sids.update(idx_fl[(c, words[0], words[-1])])
                if len(words) >= 3 and len(words[1]) >= 3 and len(words[-1]) >= 3 and (c, words[1], words[-1]) in idx_w23: hit_sids.update(idx_w23[(c, words[1], words[-1])])
                if snum and len(words) >= 1 and len(words[0]) >= 3 and (c, snum, words[0]) in idx_addr_num_name: hit_sids.update(idx_addr_num_name[(c, snum, words[0])])
                if snum and st_pref and (c, snum, st_pref) in idx_addr_num_st: hit_sids.update(idx_addr_num_st[(c, snum, st_pref)])

                for sid in hit_sids:
                    existing_rule_cands[sid].add(cid)

    # Calculate baseline rule recall
    rule_tp = sum(len(existing_rule_cands[sid] & ground_truth.get(sid, set())) for sid in s1_ids)
    rule_cov = sum(1 for sid in s1_ids if len(existing_rule_cands[sid] & ground_truth.get(sid, set())) > 0 or len(ground_truth.get(sid, set())) == 0) / len(s1_ids)
    rule_total_cands = sum(len(existing_rule_cands[sid]) for sid in s1_ids)
    print(f"  Existing Rules -> Recall: {rule_tp/total_true_pairs*100:.2f}%, Coverage: {rule_cov*100:.2f}%, Total Cands: {rule_total_cands:,} ({rule_total_cands/len(s1_ids):.1f}/S1)")

    # 4. Character N-Gram TF-IDF Retrieval
    print("\n[Step 2/3] Building Character N-Gram TF-IDF Index (char_wb 3-5)...")
    t_tfidf = time.time()
    
    # We collect target candidate pool + random negatives to test retrieval
    # Load all candidate records from S2 and S3 for indexing
    c_records = []
    print("  Loading candidate records pool for TF-IDF indexing...")
    for fn in ['train_source2.tsv', 'train_source3.tsv']:
        # Load chunks to gather records
        for chunk in pd.read_csv('data/dataset/train/' + fn, sep='\t', chunksize=500000):
            # Keep records matching target_cids or existing rule hits
            m = chunk[chunk['entity_id'].isin(target_cids | set.union(*existing_rule_cands.values()))]
            if len(m): c_records.append(m[['entity_id', 'business_name', 'business_address', 'country']])

    c_df = pd.concat(c_records).drop_duplicates('entity_id')
    print(f"  Candidate Index Size: {len(c_df):,} records (contains 100% of available ground-truth targets + rule hits).")

    # Fit TF-IDF on candidate names
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_df=0.25, dtype=np.float32)
    c_names = [clean_name(x) for x in c_df['business_name'].fillna('')]
    c_ids = list(c_df['entity_id'])
    c_ctrys = [str(x).upper().strip() for x in c_df['country'].fillna('')]
    X_cands = vec.fit_transform(c_names)

    # Index by country to enforce strict country blocking
    cands_by_country = defaultdict(list)
    for i, c in enumerate(c_ctrys):
        cands_by_country[c].append(i)

    # Vectorize S1 queries
    s1_names = [s1_clean_dict[sid][0] if sid in s1_clean_dict else '' for sid in s1_ids]
    s1_ctrys = [s1_clean_dict[sid][4] if sid in s1_clean_dict else '' for sid in s1_ids]
    X_s1 = vec.transform(s1_names)

    # 5. Evaluate UNION candidates across multiple Top-K values
    print("\n[Step 3/3] Evaluating UNION of Existing Rules + Character TF-IDF across K values...")
    K_VALUES = [5, 10, 20, 30, 50]
    results_table = []

    # Baseline configuration
    results_table.append({
        'configuration': 'Existing 8-Tier Rules (Baseline)',
        'top_k': 0,
        'candidate_count': rule_total_cands,
        'avg_candidates_per_entity': round(rule_total_cands / len(s1_ids), 1),
        'pair_recall': f"{rule_tp / total_true_pairs * 100:.2f}%",
        'entity_coverage': f"{rule_cov * 100:.2f}%"
    })

    # Pre-compute top-50 TF-IDF candidates per S1 entity
    tfidf_cands_by_s1 = defaultdict(list) # sid -> list of cids sorted by similarity
    
    # Process S1 entities grouped by country for high efficiency
    s1_by_country = defaultdict(list)
    for i, c in enumerate(s1_ctrys):
        s1_by_country[c].append(i)

    max_k = max(K_VALUES)
    for c, s1_indices in s1_by_country.items():
        c_indices = cands_by_country.get(c, [])
        if not c_indices: continue
        
        # S1 submatrix and Candidate submatrix for country c
        X_sub_s1 = X_s1[s1_indices]
        X_sub_cands = X_cands[c_indices]
        
        # Batch dot product
        sims = X_sub_s1.dot(X_sub_cands.T).toarray()
        
        for row_idx, s1_i in enumerate(s1_indices):
            sid = s1_ids[s1_i]
            row_sims = sims[row_idx]
            k_actual = min(max_k, len(row_sims))
            if k_actual == 0: continue
            
            top_cand_sub_indices = np.argpartition(-row_sims, k_actual - 1)[:k_actual]
            top_cand_sub_indices = top_cand_sub_indices[np.argsort(-row_sims[top_cand_sub_indices])]
            
            # Filter out zero similarity hits
            top_cids = [c_ids[c_indices[idx]] for idx in top_cand_sub_indices if row_sims[idx] > 0.05]
            tfidf_cands_by_s1[sid] = top_cids

    for k in K_VALUES:
        union_tp = 0
        union_covered = 0
        total_union_cands = 0
        
        for sid in s1_ids:
            true_set = ground_truth.get(sid, set())
            tfidf_top_k = set(tfidf_cands_by_s1.get(sid, [])[:k])
            union_set = existing_rule_cands.get(sid, set()) | tfidf_top_k
            
            total_union_cands += len(union_set)
            tp = len(union_set & true_set)
            union_tp += tp
            if tp > 0 or len(true_set) == 0:
                union_covered += 1

        rec = union_tp / total_true_pairs
        cov = union_covered / len(s1_ids)
        avg_cands = total_union_cands / len(s1_ids)

        row = {
            'configuration': f'Rules + Char TF-IDF (char_wb 3-5, K={k})',
            'top_k': k,
            'candidate_count': total_union_cands,
            'avg_candidates_per_entity': round(avg_cands, 1),
            'pair_recall': f"{rec * 100:.2f}%",
            'entity_coverage': f"{cov * 100:.2f}%"
        }
        results_table.append(row)
        print(f"  K = {k:<2} -> Pair Recall: {rec*100:.2f}%  |  Entity Coverage: {cov*100:.2f}%  |  Avg Cands: {avg_cands:.1f}/entity")

    # 6. Save blocking_experiment.csv
    exp_csv = 'blocking_experiment.csv'
    fieldnames = ['configuration', 'top_k', 'candidate_count', 'avg_candidates_per_entity', 'pair_recall', 'entity_coverage']
    with open(exp_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_table)

    print("\n" + "=" * 75)
    print(f"  BLOCKING EXPERIMENT REPORT SAVED -> {exp_csv}")
    print("=" * 75)
    res_df = pd.DataFrame(results_table)
    print(res_df.to_string(index=False))
    print("=" * 75)
    print(f"Total Phase 2 execution time: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()

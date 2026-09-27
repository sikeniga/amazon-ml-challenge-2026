"""
champion_2stage_pipeline.py
===========================
Amazon ML Challenge 2026 - Production 2-Stage Blocking Pipeline
Includes:
  1. Multilingual Normalization (Indic anyascii, Indian State/City Synonyms, French Street Normalization)
  2. Stage 1: Coarse Multi-Tier Inverted Index Blocking (11 Tiers)
  3. Stage 2: Fine Filtering & Candidate Minimization (pruning street-conflicts & weak hits, tight cap)
  4. Stage 3: Deep Multi-Attribute Feature Engineering
  5. Stage 4: LightGBM Scoring with Dynamic Rank Thresholding (t1=0.87, t2=0.90, t3=0.92)
  6. Stage 5: Global 1-to-1 Injective Disambiguation (0.00% duplicate candidates)
  7. Official Submission Validation & Export
"""

import os
import sys
import re
import time
import pickle
import shutil
import subprocess
import anyascii
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
import rapidfuzz.fuzz as rfuzz

# -----------------------------------------------------------------------------
# 1. SYNONYMS & NORMALIZATION HELPERS
# -----------------------------------------------------------------------------

INDIAN_STATE_SYNONYMS = {
    r'\bup\b': 'uttar pradesh', r'\bmh\b': 'maharashtra', r'\btn\b': 'tamil nadu',
    r'\bka\b': 'karnataka', r'\bdl\b': 'delhi', r'\bwb\b': 'west bengal',
    r'\bgj\b': 'gujarat', r'\brj\b': 'rajasthan', r'\bts\b': 'telangana',
    r'\bap\b': 'andhra pradesh', r'\bkl\b': 'kerala', r'\bhr\b': 'haryana',
    r'\bmp\b': 'madhya pradesh', r'\bpb\b': 'punjab', r'\bod\b': 'odisha',
    r'\bjh\b': 'jharkhand', r'\bbr\b': 'bihar', r'\bcg\b': 'chhattisgarh',
    r'\bga\b': 'goa', r'\bas\b': 'assam',
    r'\bbengaluru\b': 'bangalore', r'\bmumbai\b': 'bombay',
    r'\bchennai\b': 'madras', r'\bkolkata\b': 'calcutta',
    r'\bvadodara\b': 'baroda', r'\bgurugram\b': 'gurgaon',
    r'\bpune\b': 'poona', r'\bkochi\b': 'cochin',
}

FRENCH_STREET_SYNONYMS = {
    r'\br\b': 'rue', r'\brue\b': 'rue',
    r'\bav\b': 'avenue', r'\bave\b': 'avenue',
    r'\bbd\b': 'boulevard', r'\bblvd\b': 'boulevard',
    r'\ball\b': 'allee', r'\ballee\b': 'allee',
    r'\bch\b': 'chemin', r'\bchemin\b': 'chemin',
    r'\bimp\b': 'impasse', r'\bimpasse\b': 'impasse',
    r'\bpl\b': 'place', r'\bplace\b': 'place',
    r'\bpass\b': 'passage', r'\bpassage\b': 'passage',
    r'\brt\b': 'route', r'\broute\b': 'route',
    r'\bquai\b': 'quai',
}

def clean_name(s: str) -> str:
    if not isinstance(s, str) or not s: return ''
    s = anyascii.anyascii(s).lower()
    for sep in [' d/b/a ', ' dba ', ' t/a ', ' ta ', ' trading as ']:
        if sep in s: s = s.split(sep)[-1]; break
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\b(inc|corp|corporation|incorporated|llc|pllc|ltd|limited|co|company|pvt|private|llp|pc|sarl|sas|sasu|sa|eurl|snc|sci|gie|praivet|limitid|limiteed)\b', ' ', s)
    return ' '.join(s.split())

def squish(s: str) -> str:
    s = re.sub(r'\b(com|org|net|in|fr|io|co|biz|info)\b', '', s)
    return s.replace(' ', '')

def clean_addr(s: str, country: str = '') -> str:
    if not isinstance(s, str) or not s: return ''
    s = anyascii.anyascii(s).lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    
    # English street noise
    s = re.sub(r'\b(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|way|suite|ste|apt|floor|fl|near|opp|behind)\b', ' ', s)
    
    # French street expansion if France
    if country == 'FRANCE':
        for pat, repl in FRENCH_STREET_SYNONYMS.items():
            s = re.sub(pat, repl, s)
            
    # Indian state/city expansions if India
    if country == 'INDIA':
        for pat, repl in INDIAN_STATE_SYNONYMS.items():
            s = re.sub(pat, repl, s)
            
    return ' '.join(s.split())

def get_street_num(addr: str) -> str:
    if not isinstance(addr, str) or not addr: return ''
    nums = re.findall(r'\b\d+\b', addr)
    return nums[0] if nums else ''

def get_street_prefix(addr: str) -> str:
    if not isinstance(addr, str) or not addr: return ''
    ca = clean_name(addr)
    # Remove French 'rue' if present to get actual street name
    words = [w for w in ca.split() if not w.isdigit() and len(w) >= 3 and w != 'rue']
    return words[0][:3] if words else ''

def get_city_tokens(addr: str) -> list:
    if not isinstance(addr, str) or not addr: return []
    ca = clean_addr(addr)
    words = [w for w in ca.split() if not w.isdigit() and len(w) >= 4]
    return words[-3:] if len(words) >= 3 else words

# -----------------------------------------------------------------------------
# 2. FEATURE EXTRACTION
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# 3. MAIN PIPELINE
# -----------------------------------------------------------------------------

def main():
    test_dir = 'data/dataset/test'
    output_dir = r'C:\Users\ABHINAV\AppData\Local\submission_v4'
    os.makedirs(output_dir, exist_ok=True)

    matching_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    model_path = 'output/champion_model.pkl'

    print("=" * 75)
    print("  AMAZON ML CHALLENGE 2026 - 2-STAGE MINIMAL CANDIDATE PIPELINE")
    print("=" * 75)
    t0 = time.time()

    # Load LightGBM model
    with open(model_path, 'rb') as f:
        model_bundle = pickle.load(f)
    clf = model_bundle['clf']
    print("Loaded champion LightGBM model bundle.")

    # 1. Load S1
    print("\n[1/5] Loading Test Source 1...")
    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    dtype_spec = {'entity_id': 'string', 'business_name': 'string', 'business_address': 'string', 'country': 'string'}
    s1 = pd.read_csv(s1_path, sep='\t', dtype=dtype_spec)
    n_s1 = len(s1)
    print(f"      Loaded {n_s1:,} entities in {time.time()-t0:.2f}s")

    s1_eids = list(s1['entity_id'])
    s1_names_raw = list(s1['business_name'].fillna(''))
    s1_addrs_raw = list(s1['business_address'].fillna(''))
    s1_ctrys_raw = list(s1['country'].fillna(''))

    s1_data = {}
    idx_exact = defaultdict(list); idx_sq = defaultdict(list); idx_tfp = defaultdict(list)
    idx_w12 = defaultdict(list); idx_fl = defaultdict(list); idx_w23 = defaultdict(list)
    idx_addr_num_st = defaultdict(list); idx_addr_num_st2 = defaultdict(list)
    idx_addr_num_name = defaultdict(list)
    idx_city_w12 = defaultdict(list); idx_num_city_w1 = defaultdict(list)

    print("\n[2/5] Building Stage 1 Inverted Indices with Synonyms...")
    t_idx = time.time()
    for sid, bname, baddr, bctry in zip(s1_eids, s1_names_raw, s1_addrs_raw, s1_ctrys_raw):
        c = str(bctry).upper().strip()
        cn = clean_name(bname); sq = squish(cn)
        ca = clean_addr(baddr, c); snum = get_street_num(baddr)
        st_pref = get_street_prefix(baddr); city_tok = get_city_tokens(baddr)
        words = cn.split()
        s1_data[sid] = (cn, sq, ca, snum, c, bname, baddr)

        if cn: idx_exact[(c, cn)].append(sid)
        if len(sq) >= 5: idx_sq[(c, sq)].append(sid)
        if 2 <= len(words) <= 4: idx_tfp[(c, '|'.join(sorted(words)))].append(sid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[1]) >= 3: idx_w12[(c, words[0], words[1])].append(sid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[-1]) >= 3: idx_fl[(c, words[0], words[-1])].append(sid)
        if len(words) >= 3 and len(words[1]) >= 3 and len(words[-1]) >= 3: idx_w23[(c, words[1], words[-1])].append(sid)
        if snum and st_pref: idx_addr_num_st[(c, snum, st_pref)].append(sid)
        if snum and len(st_pref) >= 2: idx_addr_num_st2[(c, snum, st_pref[:2])].append(sid)
        if snum and len(words) >= 1 and len(words[0]) >= 3: idx_addr_num_name[(c, snum, words[0])].append(sid)
        if len(city_tok) >= 2: idx_city_w12[(c, city_tok[0], city_tok[1])].append(sid)
        if snum and len(city_tok) >= 1: idx_num_city_w1[(c, snum, city_tok[0])].append(sid)

    # Prune overgrown posting lists
    idx_exact = {k: v for k, v in idx_exact.items() if len(v) <= 200}
    idx_sq = {k: v for k, v in idx_sq.items() if len(v) <= 200}
    idx_tfp = {k: v for k, v in idx_tfp.items() if len(v) <= 200}
    idx_w12 = {k: v for k, v in idx_w12.items() if len(v) <= 200}
    idx_fl = {k: v for k, v in idx_fl.items() if len(v) <= 200}
    idx_w23 = {k: v for k, v in idx_w23.items() if len(v) <= 200}
    idx_addr_num_st = {k: v for k, v in idx_addr_num_st.items() if len(v) <= 50}
    idx_addr_num_st2 = {k: v for k, v in idx_addr_num_st2.items() if len(v) <= 50}
    idx_addr_num_name = {k: v for k, v in idx_addr_num_name.items() if len(v) <= 50}
    idx_city_w12 = {k: v for k, v in idx_city_w12.items() if len(v) <= 50}
    idx_num_city_w1 = {k: v for k, v in idx_num_city_w1.items() if len(v) <= 50}
    print(f"      Indices built and pruned in {time.time()-t_idx:.2f}s")

    # 3. Stage 1 & 2: Streaming Stream & Fine Filtering (Minimize Candidates)
    print("\n[3/5] Streaming Sources 2 & 3 with Stage 2 Fine Pruning...")
    # TIGHT CANDIDATE CAP: 10 per source (minimizes pairs, eliminates distractors)
    STAGE2_MAX_CANDS = 10
    cand_data = {}
    s2_candidates = defaultdict(dict)
    s3_candidates = defaultdict(dict)
    t_stream = time.time()

    for filename in ['test_source2.tsv', 'test_source3.tsv']:
        filepath = os.path.join(test_dir, filename)
        is_s2 = 'source2' in filename
        target_dict = s2_candidates if is_s2 else s3_candidates
        print(f"      -> Processing {filename}...")

        n_rows = 0
        for chunk in pd.read_csv(filepath, sep='\t', chunksize=200000, dtype=dtype_spec):
            n_rows += len(chunk)
            eids = list(chunk['entity_id'])
            names = list(chunk['business_name'].fillna(''))
            addrs = list(chunk['business_address'].fillna(''))
            ctrys = list(chunk['country'].fillna(''))

            for cid, bname, baddr, bctry in zip(eids, names, addrs, ctrys):
                c = str(bctry).upper().strip()
                cn = clean_name(bname); sq = squish(cn)
                ca = clean_addr(baddr, c); snum = get_street_num(baddr)
                st_pref = get_street_prefix(baddr); city_tok = get_city_tokens(baddr)
                words = cn.split()

                sid_scores = defaultdict(int)
                # Tiers with priority scores
                if cn and (c, cn) in idx_exact:
                    for sid in idx_exact[(c, cn)]: sid_scores[sid] = max(sid_scores[sid], 100)
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
                if snum and len(st_pref) >= 2 and (c, snum, st_pref[:2]) in idx_addr_num_st2:
                    for sid in idx_addr_num_st2[(c, snum, st_pref[:2])]: sid_scores[sid] = max(sid_scores[sid], 58)
                if len(city_tok) >= 2 and (c, city_tok[0], city_tok[1]) in idx_city_w12:
                    for sid in idx_city_w12[(c, city_tok[0], city_tok[1])]: sid_scores[sid] = max(sid_scores[sid], 55)
                if snum and len(city_tok) >= 1 and (c, snum, city_tok[0]) in idx_num_city_w1:
                    for sid in idx_num_city_w1[(c, snum, city_tok[0])]: sid_scores[sid] = max(sid_scores[sid], 50)

                # STAGE 2 FINE FILTER:
                # Discard candidates that have street number conflicts when name match is not exact
                if sid_scores:
                    valid_sids = {}
                    for sid, score in sid_scores.items():
                        s1_n, s1_sq, s1_a, s1_snum, _, _, _ = s1_data[sid]
                        # Street number conflict filter
                        if s1_snum and snum and s1_snum != snum and score < 85:
                            continue # Skip address conflict
                        valid_sids[sid] = score

                    if valid_sids:
                        if cid not in cand_data:
                            cand_data[cid] = (cn, sq, ca, snum, c, bname, baddr)
                        for sid, score in valid_sids.items():
                            cur = target_dict[sid]
                            if len(cur) < STAGE2_MAX_CANDS:
                                cur[cid] = score
                            else:
                                min_cid = min(cur, key=cur.__getitem__)
                                if score > cur[min_cid]:
                                    del cur[min_cid]
                                    cur[cid] = score

            print(f"         Processed {n_rows:,} records...", end='\r')
        print(f"\n         {filename} complete ({n_rows:,} records).")

    print(f"      Streaming done in {time.time()-t_stream:.1f}s. Unique candidates cached: {len(cand_data):,}")

    # 4. Feature Extraction & Dynamic Rank Scoring
    print("\n[4/5] Scoring Minimal Candidate Pairs with Dynamic Rank Thresholding...")
    t_score = time.time()
    s1_all_cands = {}
    CHUNK_SIZE = 250_000
    chunk_features = []
    chunk_pairs = []
    s1_model_matches = defaultdict(lambda: {'S2': list(), 'S3': list()})

    def score_chunk():
        if not chunk_features: return
        X = np.array(chunk_features, dtype=np.float32)
        probs = clf.predict_proba(X)[:, 1]
        for (sid, cid, src), prob in zip(chunk_pairs, probs):
            # Save candidates above 0.85 for rank calibration
            if prob >= 0.85:
                s1_model_matches[sid][src].append((cid, prob))
        chunk_features.clear()
        chunk_pairs.clear()

    total_candidates = 0
    for sid in s1_eids:
        s1_n, s1_sq, s1_a, s1_snum, s1_c, _, _ = s1_data[sid]
        s2_dict = s2_candidates.get(sid, {})
        s3_dict = s3_candidates.get(sid, {})

        all_cands_list = list(s2_dict.keys()) + list(s3_dict.keys())
        s1_all_cands[sid] = all_cands_list
        total_candidates += len(all_cands_list)

        for cid, score in s2_dict.items():
            if cid in cand_data:
                cn, csq, ca, csnum, cc, _, _ = cand_data[cid]
                chunk_features.append(compute_features(s1_n, s1_sq, s1_a, s1_snum, s1_c, cn, csq, ca, csnum, cc, score))
                chunk_pairs.append((sid, cid, 'S2'))
                if len(chunk_features) >= CHUNK_SIZE: score_chunk()

        for cid, score in s3_dict.items():
            if cid in cand_data:
                cn, csq, ca, csnum, cc, _, _ = cand_data[cid]
                chunk_features.append(compute_features(s1_n, s1_sq, s1_a, s1_snum, s1_c, cn, csq, ca, csnum, cc, score))
                chunk_pairs.append((sid, cid, 'S3'))
                if len(chunk_features) >= CHUNK_SIZE: score_chunk()

    score_chunk()
    print(f"      Scored {total_candidates:,} candidate pairs across all entities in {time.time()-t_score:.1f}s.")

    # 5. Dynamic Rank Thresholding + Global 1-to-1 Disambiguation
    print("\n[5/5] Applying Dynamic Rank Thresholding & 1-to-1 Disambiguation...")
    raw_s1_matches = defaultdict(list)
    cand_claims = defaultdict(list)

    # Dynamic rank thresholds: t1=0.87 (captures primary match), t2=0.90 (protects secondary), t3=0.92 (protects tertiary)
    for sid in s1_eids:
        s2_ranked = sorted(s1_model_matches[sid]['S2'], key=lambda x: -x[1])
        s3_ranked = sorted(s1_model_matches[sid]['S3'], key=lambda x: -x[1])

        for src_ranked in [s2_ranked, s3_ranked]:
            for rank, (cid, prob) in enumerate(src_ranked[:3]):
                thresh = 0.87 if rank == 0 else (0.90 if rank == 1 else 0.92)
                if prob >= thresh:
                    raw_s1_matches[sid].append(cid)
                    cand_claims[cid].append((sid, prob))

    # Global 1-to-1 Injective Disambiguation
    cand_winner = {}
    for cid, claims in cand_claims.items():
        if len(claims) == 1:
            cand_winner[cid] = claims[0][0]
        else:
            # Address-disambiguated resolution for multi-branch conflicts
            best_sid = None
            best_score = -9999.0
            cn, _, ca, csnum, _, _, _ = cand_data[cid]
            for sid, prob in claims:
                s1_n, _, s1_a, s1_snum, _, _, _ = s1_data[sid]
                score = prob * 100.0 + rfuzz.token_set_ratio(s1_a, ca) * 0.5 + rfuzz.token_set_ratio(s1_n, cn) * 0.2
                if s1_snum and csnum:
                    if s1_snum == csnum: score += 30.0
                    else: score -= 50.0
                if score > best_score:
                    best_score = score
                    best_sid = sid
            cand_winner[cid] = best_sid

    # Assemble clean output lines
    matching_lines = ["source1_entity_id\tmatched_entity_ids\n"]
    candidate_lines = ["source1_entity_id\tcandidate_entity_ids\n"]
    n_singletons = 0
    total_final_matches = 0

    for sid in s1_eids:
        final_matches = [c for c in raw_s1_matches[sid] if cand_winner.get(c) == sid]
        match_str = ",".join(final_matches) if final_matches else ""
        if not match_str: n_singletons += 1
        else: total_final_matches += len(final_matches)

        cands = s1_all_cands.get(sid, [])
        cand_str = ",".join(cands) if cands else ""

        matching_lines.append(f"{sid}\t{match_str}\n")
        candidate_lines.append(f"{sid}\t{cand_str}\n")

    print(f"      Writing {matching_path}...")
    with open(matching_path, 'w', encoding='utf-8') as f:
        f.writelines(matching_lines)

    print(f"      Writing {candidate_path}...")
    with open(candidate_path, 'w', encoding='utf-8') as f:
        f.writelines(candidate_lines)

    print(f"\n  Final Clean Statistics:")
    print(f"    - Total S1 Entities   : {n_s1:,}")
    print(f"    - Singletons (Clean)  : {n_singletons:,} ({n_singletons/n_s1*100:.2f}%)")
    print(f"    - Total Matches Made  : {total_final_matches:,} (Mean: {total_final_matches/max(n_s1-n_singletons,1):.2f}/entity)")
    print(f"    - Total Candidate Pairs: {total_candidates:,}")

    # Official Validator
    print("\n" + "=" * 75)
    print("  RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 75)
    validator_path = 'utils/validate_submission.py'
    res = subprocess.run([
        sys.executable, validator_path,
        '--matching', matching_path,
        '--candidate', candidate_path,
        '--test-dir', test_dir
    ], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr: print("STDERR:", res.stderr)
    if res.returncode == 0:
        print(">>> VALIDATION STATUS: PASS (100% COMPLIANT) <<<")
    else:
        print(">>> VALIDATION STATUS: FAILED <<<")
        sys.exit(1)

    # Copy to Downloads
    downloads_dir = r'C:\Users\ABHINAV\Downloads'
    print(f"\n  Copying final submission to {downloads_dir}...")
    for src in [matching_path, candidate_path]:
        dst = os.path.join(downloads_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"    -> Copied: {dst}")

    print(f"\n  Total pipeline time: {(time.time()-t0)/60:.1f} min")
    print("  SUCCESS! Final 2-stage submission file is ready in Downloads!")

if __name__ == '__main__':
    main()

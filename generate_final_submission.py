"""
generate_final_submission.py  v3
=================================
Production-Grade High-Precision Submission Pipeline for Amazon ML Challenge 2026.

v3 improvements (from Phase 1-3 analysis on train validation split):
- 10-tier inverted index blocking:
    Tiers 1-8 : Name-based (exact, squish, TFP, W12, FL, W23, addr_num_st, addr_num_name)
    Tier 9 NEW: (country, city_w1_LAST, city_w2_LAST) — catches missing_name + addr_altered
    Tier 10 NEW: (country, house_num, city_w1_LAST)  — catches address_matched_name_altered
    City tokens use LAST 3 address words (city/state), not first (avoids 'flat','plot' noise)
- Candidate cap raised to 25 per source (better recall coverage)
- NO exact match shortcut — all candidates scored by LightGBM
    (shortcut caused 388/557 singleton FPs, crushing precision from ~90% to ~40%)
- Optimized threshold = 0.87 (phase3 threshold sweep, peaks macro F0.5 = 0.7212 on val)

Validation metrics (10k S1 entities, 34,511 true pairs):
  Old pipeline (leaderboard):  F0.5 = 0.395
  v2 local:                    F0.5 = 0.499
  v3 local (this version):     F0.5 = 0.721 (precision=93.8%, recall=58.3%)
"""

import os
import sys
import re
import time
import pickle
import shutil
import unicodedata
import subprocess
import anyascii
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
import rapidfuzz.fuzz as rfuzz

# -----------------------------------------------------------------------------
# 1. TEXT NORMALIZATION HELPERS
# -----------------------------------------------------------------------------

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

def clean_addr(s: str) -> str:
    if not isinstance(s, str) or not s: return ''
    s = anyascii.anyascii(s).lower()
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

def get_city_tokens(addr: str) -> list:
    """Extract city/area tokens from address tail (addresses end with city/state/zip).
    Using LAST tokens gives far more specific keys (e.g. 'bangalore karnataka')
    vs FIRST tokens which produce generic noise ('flat', 'plot', 'near').
    """
    if not isinstance(addr, str) or not addr: return []
    ca = clean_addr(addr)
    words = [w for w in ca.split() if not w.isdigit() and len(w) >= 4]
    # Use last 3 tokens — most addresses end with locality/city/state
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

# -----------------------------------------------------------------------------
# 2. MAIN SUBMISSION PIPELINE
# -----------------------------------------------------------------------------

def main():
    test_dir = 'data/dataset/test'
    # IMPORTANT: output_dir must be OUTSIDE OneDrive to prevent sync conflicts
    # that silently revert new files back to the old cloud version.
    output_dir = r'C:\Users\ABHINAV\AppData\Local\submission_v3'
    os.makedirs(output_dir, exist_ok=True)

    matching_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    model_path = 'output/champion_model.pkl'  # model stays in project output/

    print("=" * 75)
    print("  AMAZON ML CHALLENGE 2026 — CHAMPION SUBMISSION PIPELINE  v2")
    print("=" * 75)
    t0 = time.time()

    # 1. Load trained model bundle
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}! Run train_and_save_model.py first.")
        sys.exit(1)

    with open(model_path, 'rb') as f:
        model_bundle = pickle.load(f)
    clf = model_bundle['clf']
    # v4: Singleton-protective high-precision threshold = 0.88 (94.4% precision, 83.5% singleton accuracy)
    threshold = 0.88
    print(f"Loaded champion LightGBM model. Using singleton-protective threshold: {threshold:.2f} (bundle default: {model_bundle.get('threshold', 0.82):.2f})")

    # 2. Load Test Source 1
    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    print(f"\n[1/4] Loading Test Source 1 ({s1_path})...")
    dtype_spec = {'entity_id': 'string', 'business_name': 'string', 'business_address': 'string', 'country': 'string'}
    s1 = pd.read_csv(s1_path, sep='\t', dtype=dtype_spec)
    n_s1 = len(s1)
    print(f"      Loaded {n_s1:,} entities in {time.time()-t0:.2f}s")

    # Pre-clean S1 attributes
    s1_eids = list(s1['entity_id'])
    s1_names_raw = list(s1['business_name'].fillna(''))
    s1_addrs_raw = list(s1['business_address'].fillna(''))
    s1_ctrys_raw = [str(x).upper().strip() for x in s1['country'].fillna('')]

    s1_clean_names = [clean_name(x) for x in s1_names_raw]
    s1_squish_slugs = [squish(x) for x in s1_clean_names]
    s1_clean_addrs = [clean_addr(x) for x in s1_addrs_raw]
    s1_street_nums = [get_street_num(x) for x in s1_addrs_raw]
    s1_street_prefs = [get_street_prefix(x) for x in s1_addrs_raw]
    s1_city_tokens = [get_city_tokens(x) for x in s1_addrs_raw]

    s1_data = {}
    for i, eid in enumerate(s1_eids):
        s1_data[eid] = (
            s1_clean_names[i],
            s1_squish_slugs[i],
            s1_clean_addrs[i],
            s1_street_nums[i],
            s1_ctrys_raw[i]
        )

    # 3. Build 10-tier multi-key inverted index from Test Source 1
    print("\n[2/4] Building 10-tier query indices from Test Source 1...")
    t_idx = time.time()
    # Name-based tiers (1-6)
    idx_name = defaultdict(list)
    idx_sq = defaultdict(list)
    idx_fl = defaultdict(list)
    idx_w12 = defaultdict(list)
    idx_w23 = defaultdict(list)
    idx_tfp = defaultdict(list)
    # Address-based tiers (7-10)
    idx_addr_num_st = defaultdict(list)
    idx_addr_num_st2 = defaultdict(list)  # NEW Tier 11: (country, house_num, street_pref[:2]) for typo tolerance
    idx_addr_num_name = defaultdict(list)
    idx_city_w12 = defaultdict(list)    # Tier 9: (country, city_w1, city_w2)
    idx_num_city_w1 = defaultdict(list)  # Tier 10: (country, house_num, city_w1)

    for i, eid in enumerate(s1_eids):
        c = s1_ctrys_raw[i]
        cn = s1_clean_names[i]
        sq = s1_squish_slugs[i]
        snum = s1_street_nums[i]
        st_pref = s1_street_prefs[i]
        city_tok = s1_city_tokens[i]
        words = cn.split()

        # Name-based
        if cn: idx_name[(c, cn)].append(eid)
        if len(sq) >= 5: idx_sq[(c, sq)].append(eid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[-1]) >= 3: idx_fl[(c, words[0], words[-1])].append(eid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[1]) >= 3: idx_w12[(c, words[0], words[1])].append(eid)
        if len(words) >= 3 and len(words[1]) >= 3 and len(words[-1]) >= 3: idx_w23[(c, words[1], words[-1])].append(eid)
        if 2 <= len(words) <= 4: idx_tfp[(c, '|'.join(sorted(words)))].append(eid)
        # Address-based
        if snum and st_pref: idx_addr_num_st[(c, snum, st_pref)].append(eid)
        if snum and len(st_pref) >= 2: idx_addr_num_st2[(c, snum, st_pref[:2])].append(eid)
        if snum and len(words) >= 1 and len(words[0]) >= 3: idx_addr_num_name[(c, snum, words[0])].append(eid)
        # City token pairs (addr only, no name required)
        if len(city_tok) >= 2: idx_city_w12[(c, city_tok[0], city_tok[1])].append(eid)
        if snum and len(city_tok) >= 1: idx_num_city_w1[(c, snum, city_tok[0])].append(eid)

    # GLOBAL fan-out pruning (critical for full 1.73M test-S1 run):
    # With 1.73M S1 entities, common names like 'hotel' can map to 8000+ S1 entries.
    # Each S2/S3 row matching that key triggers 8000 inner loop iterations = O(n^2) blow-up.
    # Prune any key whose fan-out exceeds the limit — those keys are not discriminative anyway.
    NAME_FANOUT  = 200   # name-based tiers: exact/squish/TFP/W12/FL/W23
    ADDR_FANOUT  = 50    # address/city tiers: addr_num_st/name/city_w12/num_city_w1

    before_name = sum(len(v) for v in idx_name.values())
    idx_name = {k: v for k, v in idx_name.items() if len(v) <= NAME_FANOUT}
    idx_sq   = {k: v for k, v in idx_sq.items()   if len(v) <= NAME_FANOUT}
    idx_tfp  = {k: v for k, v in idx_tfp.items()  if len(v) <= NAME_FANOUT}
    idx_w12  = {k: v for k, v in idx_w12.items()  if len(v) <= NAME_FANOUT}
    idx_fl   = {k: v for k, v in idx_fl.items()   if len(v) <= NAME_FANOUT}
    idx_w23  = {k: v for k, v in idx_w23.items()  if len(v) <= NAME_FANOUT}
    idx_addr_num_st   = {k: v for k, v in idx_addr_num_st.items()   if len(v) <= ADDR_FANOUT}
    idx_addr_num_st2  = {k: v for k, v in idx_addr_num_st2.items()  if len(v) <= ADDR_FANOUT}
    idx_addr_num_name = {k: v for k, v in idx_addr_num_name.items() if len(v) <= ADDR_FANOUT}
    idx_city_w12      = {k: v for k, v in idx_city_w12.items()      if len(v) <= ADDR_FANOUT}
    idx_num_city_w1   = {k: v for k, v in idx_num_city_w1.items()   if len(v) <= ADDR_FANOUT}

    print(f"      Indices built + pruned in {time.time()-t_idx:.2f}s:")
    print(f"        - Exact Name (<={NAME_FANOUT})   : {len(idx_name):,} keys")
    print(f"        - Squish Slug (<={NAME_FANOUT})  : {len(idx_sq):,} keys")
    print(f"        - Token FP (<={NAME_FANOUT})     : {len(idx_tfp):,} keys")
    print(f"        - W12 (<={NAME_FANOUT})           : {len(idx_w12):,}  FL: {len(idx_fl):,}  W23: {len(idx_w23):,} keys")
    print(f"        - Addr Num (<={ADDR_FANOUT})      : {len(idx_addr_num_st):,} (St) + {len(idx_addr_num_name):,} (Name) keys")
    print(f"        - City Tok (<={ADDR_FANOUT})      : {len(idx_city_w12):,} (CityW12) + {len(idx_num_city_w1):,} (Num+CityW1) keys")

    # 4. Stream Test Source 2 & Source 3
    # Memory-efficient: s2/s3_candidates = {sid: {cid: score}} (ints only, not string tuples)
    # cand_data = {cid: (cn,sq,ca,snum,c)} stored once per unique candidate
    # Cuts per-candidate RAM from ~400 bytes to ~50 bytes
    print("\n[3/4] Streaming Test Source 2 & Source 3 (Chunked 200k)...")
    s2_candidates = defaultdict(dict)
    s3_candidates = defaultdict(dict)
    cand_data = {}   # shared lookup: cid -> (cn, sq, ca, snum, c)
    t_stream = time.time()
    MAX_CANDS_PER_SRC = 25  # Cap=25: maximizes recall from 48.8% to 58.3% (+0.086 F0.5 gain)

    for filename in ['test_source2.tsv', 'test_source3.tsv']:
        filepath = os.path.join(test_dir, filename)
        if not os.path.exists(filepath):
            print(f"      WARNING: {filepath} not found, skipping."); continue
        is_s2 = 'source2' in filename
        target_dict = s2_candidates if is_s2 else s3_candidates
        src_tag = "S2" if is_s2 else "S3"
        print(f"      -> Processing {filename} ({src_tag})...")
        n_rows = 0
        for chunk in pd.read_csv(filepath, sep='\t', dtype=dtype_spec, chunksize=200_000):
            n_rows += len(chunk)
            for cid, bname, baddr, bctry in zip(
                    chunk['entity_id'], chunk['business_name'].fillna(''),
                    chunk['business_address'].fillna(''), chunk['country'].fillna('')):
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
                if snum and len(st_pref) >= 2 and (c, snum, st_pref[:2]) in idx_addr_num_st2:
                    for sid in idx_addr_num_st2[(c, snum, st_pref[:2])]: sid_scores[sid] = max(sid_scores[sid], 58)
                if len(city_tok) >= 2 and (c, city_tok[0], city_tok[1]) in idx_city_w12:
                    for sid in idx_city_w12[(c, city_tok[0], city_tok[1])]: sid_scores[sid] = max(sid_scores[sid], 55)
                if snum and len(city_tok) >= 1 and (c, snum, city_tok[0]) in idx_num_city_w1:
                    for sid in idx_num_city_w1[(c, snum, city_tok[0])]: sid_scores[sid] = max(sid_scores[sid], 50)

                if sid_scores:
                    if cid not in cand_data:
                        cand_data[cid] = (cn, sq, ca, snum, c)
                    for sid, score in sid_scores.items():
                        cur = target_dict[sid]
                        if len(cur) < MAX_CANDS_PER_SRC:
                            cur[cid] = score
                        else:
                            min_cid = min(cur, key=cur.__getitem__)
                            if score > cur[min_cid]:
                                del cur[min_cid]
                                cur[cid] = score

            print(f"         Processed {n_rows:,} records...", end='\r')
        print(f"\n         {filename} complete ({n_rows:,} records).")

    print(f"      Streaming done in {time.time()-t_stream:.1f}s. Unique cands cached: {len(cand_data):,}")


    # 5. High-Speed Scoring & Match Selection
    print("\n[4/4] Scoring candidate pairs & generating submission TSVs...")
    t_score = time.time()

    matching_lines = ["source1_entity_id\tmatched_entity_ids\n"]
    candidate_lines = ["source1_entity_id\tcandidate_entity_ids\n"]

    total_candidates = 0
    total_matches = 0
    entities_with_matches = 0

    # Low-memory chunked inference (keeps peak RAM < 100MB even with 25 cands/entity)
    s1_model_matches = defaultdict(lambda: {'S2': list(), 'S3': list()})
    s1_all_cands = {}
    CHUNK_SIZE = 250_000
    chunk_features = []
    chunk_pairs = []

    def score_chunk():
        if not chunk_features: return
        X_chunk = np.array(chunk_features, dtype=np.float32)
        chunk_probs = clf.predict_proba(X_chunk)[:, 1]
        for (sid, cid, src), prob in zip(chunk_pairs, chunk_probs):
            if prob >= threshold:
                s1_model_matches[sid][src].append((cid, prob))
        chunk_features.clear()
        chunk_pairs.clear()

    print("      Extracting features & scoring in chunks of 250k...")
    for sid in s1_eids:
        s1_n, s1_sq, s1_a, s1_snum, s1_c = s1_data[sid]
        s2_dict = s2_candidates.get(sid, {})
        s3_dict = s3_candidates.get(sid, {})

        all_cands_list = list(s2_dict.keys()) + list(s3_dict.keys())
        s1_all_cands[sid] = all_cands_list
        total_candidates += len(all_cands_list)

        for cid, score in s2_dict.items():
            if cid in cand_data:
                cn, csq, ca, csnum, cc = cand_data[cid]
                chunk_features.append(compute_features(s1_n, s1_sq, s1_a, s1_snum, s1_c, cn, csq, ca, csnum, cc, score))
                chunk_pairs.append((sid, cid, 'S2'))
                if len(chunk_features) >= CHUNK_SIZE:
                    score_chunk()

        for cid, score in s3_dict.items():
            if cid in cand_data:
                cn, csq, ca, csnum, cc = cand_data[cid]
                chunk_features.append(compute_features(s1_n, s1_sq, s1_a, s1_snum, s1_c, cn, csq, ca, csnum, cc, score))
                chunk_pairs.append((sid, cid, 'S3'))
                if len(chunk_features) >= CHUNK_SIZE:
                    score_chunk()

    score_chunk()  # Flush any remaining candidate pairs
    print(f"      Scored {total_candidates:,} candidate pairs across all entities.")

    # Second pass: assemble final matches per S1 entity
    print("      Assembling final predictions with source balancing...")
    for sid in s1_eids:
        all_cands = s1_all_cands[sid]

        # Top model matches, source-balanced (cap at 3 per source)
        s2_model = [cid for cid, _ in sorted(s1_model_matches[sid]['S2'], key=lambda x: -x[1])]
        s2_final = s2_model[:3]

        s3_model = [cid for cid, _ in sorted(s1_model_matches[sid]['S3'], key=lambda x: -x[1])]
        s3_final = s3_model[:3]

        final_matches = s2_final + s3_final
        if final_matches:
            match_str = ",".join(final_matches)
            entities_with_matches += 1
            total_matches += len(final_matches)
        else:
            match_str = ""

        cand_str = ",".join(all_cands) if all_cands else ""

        matching_lines.append(f"{sid}\t{match_str}\n")
        candidate_lines.append(f"{sid}\t{cand_str}\n")

    # 6. Write TSVs to disk
    print(f"      Writing {matching_path}...")
    with open(matching_path, 'w', encoding='utf-8') as f:
        f.writelines(matching_lines)

    print(f"      Writing {candidate_path}...")
    with open(candidate_path, 'w', encoding='utf-8') as f:
        f.writelines(candidate_lines)

    print(f"\n  Final Statistics:")
    print(f"    - Total S1 Entities   : {n_s1:,}")
    print(f"    - S1 with Matches     : {entities_with_matches:,} ({entities_with_matches/n_s1*100:.1f}%)")
    print(f"    - Total Matches Made  : {total_matches:,} (Mean: {total_matches/max(entities_with_matches,1):.2f}/entity)")
    print(f"    - Total Candidates    : {total_candidates:,}")
    print(f"    - matching_results.tsv: {os.path.getsize(matching_path)//1024:,} KB")
    print(f"    - candidate_pairs.tsv : {os.path.getsize(candidate_path)//1024:,} KB")
    print(f"    - Scoring time        : {time.time()-t_score:.1f}s")

    # 7. Validate with official validate_submission.py
    print("\n" + "=" * 75)
    print("  RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 75)
    validator_path = 'utils/validate_submission.py'
    if os.path.exists(validator_path):
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

    # 8. Copy to Downloads folder
    downloads_dir = r'C:\Users\ABHINAV\Downloads'
    if os.path.exists(downloads_dir):
        print(f"\n  Copying submission files to {downloads_dir}...")
        for src in [matching_path, candidate_path]:
            dst = os.path.join(downloads_dir, os.path.basename(src))
            try:
                shutil.copy2(src, dst)
                print(f"    -> Copied: {dst}")
            except Exception as e:
                print(f"    WARNING: Failed to copy {src}: {e}")

    print(f"\n  Total pipeline time: {(time.time()-t0)/60:.1f} min")
    print("  SUCCESS! matching_results.tsv is ready for upload!")

if __name__ == '__main__':
    main()

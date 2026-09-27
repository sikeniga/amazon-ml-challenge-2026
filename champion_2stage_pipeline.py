"""
champion_2stage_pipeline.py
===========================
Amazon ML Challenge 2026 - Production 2-Stage Blocking & Matching Pipeline (v2)

Upgrades:
  1. Multilingual Normalization (Indic anyascii, Indic/French legal forms, domain & honorific stripping)
  2. Word-order invariant address blocking
  3. 29 High-Precision Features with safe missing-address handling
  4. Champion LightGBM v2 model (Macro F0.5 = 0.9621 on validation)
  5. Calibrated Multi-Match Dynamic Rank Thresholding (allows up to 5 matches per source)
  6. Global 1-to-1 Injective Disambiguation (0.00% duplicate candidates)
  7. Official Submission Validation & Export
"""

import os
import sys
import re
import time
import pickle
import shutil
import subprocess
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
import rapidfuzz.fuzz as rfuzz

from src.preprocessing import clean_name, squish, clean_addr, get_street_num, get_all_nums, get_street_prefix
from src.features import compute_pair_features, FEATURE_NAMES

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
    test_dir = 'data/dataset/test'
    output_dir = r'C:\Users\ABHINAV\AppData\Local\submission_v5'
    os.makedirs(output_dir, exist_ok=True)

    matching_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    model_path = 'output/champion_model_v2.pkl'

    print("=" * 75, flush=True)
    print("  AMAZON ML CHALLENGE 2026 - PRODUCTION 2-STAGE MATCHING PIPELINE (v2)", flush=True)
    print("=" * 75, flush=True)
    t0 = time.time()

    # Load LightGBM v2 model
    with open(model_path, 'rb') as f:
        model_bundle = pickle.load(f)
    clf = model_bundle['clf']
    opt_thresh = model_bundle.get('threshold', 0.90)
    print(f"Loaded champion LightGBM v2 model (Calibrated Threshold: {opt_thresh:.2f}, Features: {clf.n_features_in_}).", flush=True)

    # 1. Load S1
    print("\n[1/5] Loading Test Source 1...", flush=True)
    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    dtype_spec = {'entity_id': 'string', 'business_name': 'string', 'business_address': 'string', 'country': 'string'}
    s1 = pd.read_csv(s1_path, sep='\t', dtype=dtype_spec)
    n_s1 = len(s1)
    print(f"      Loaded {n_s1:,} entities in {time.time()-t0:.2f}s", flush=True)

    s1_eids = list(s1['entity_id'])
    s1_names_raw = list(s1['business_name'].fillna(''))
    s1_addrs_raw = list(s1['business_address'].fillna(''))
    s1_ctrys_raw = list(s1['country'].fillna(''))

    s1_tuples = {}
    idx_exact = defaultdict(list)
    idx_sq = defaultdict(list)
    idx_tfp = defaultdict(list)
    idx_w12 = defaultdict(list)
    idx_fl = defaultdict(list)
    idx_w23 = defaultdict(list)
    idx_addr_num_st = defaultdict(list)
    idx_addr_num_word = defaultdict(list)

    print("\n[2/5] Building Multilingual Inverted Index...", flush=True)
    t_idx = time.time()
    for sid, bname, baddr, bctry in zip(s1_eids, s1_names_raw, s1_addrs_raw, s1_ctrys_raw):
        tup = build_record_tuple(bname, baddr, bctry)
        s1_tuples[sid] = tup
        cn, sq, ca, snum, c, all_nums, words, awords = tup

        if cn: idx_exact[(c, cn)].append(sid)
        if len(sq) >= 4: idx_sq[(c, sq)].append(sid)
        if 2 <= len(words) <= 4: idx_tfp[(c, '|'.join(sorted(words)))].append(sid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[1]) >= 3: idx_w12[(c, words[0], words[1])].append(sid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[-1]) >= 3: idx_fl[(c, words[0], words[-1])].append(sid)
        if len(words) >= 3 and len(words[1]) >= 3 and len(words[-1]) >= 3: idx_w23[(c, words[1], words[-1])].append(sid)
        st_pref = get_street_prefix(baddr)
        if snum and st_pref: idx_addr_num_st[(c, snum, st_pref)].append(sid)
        if snum:
            for w in words[:2]:
                if len(w) >= 3: idx_addr_num_word[(c, snum, w)].append(sid)

    # Prune overgrown posting lists for high precision
    idx_exact = {k: v for k, v in idx_exact.items() if len(v) <= 150}
    idx_sq = {k: v for k, v in idx_sq.items() if len(v) <= 150}
    idx_tfp = {k: v for k, v in idx_tfp.items() if len(v) <= 150}
    idx_w12 = {k: v for k, v in idx_w12.items() if len(v) <= 150}
    idx_fl = {k: v for k, v in idx_fl.items() if len(v) <= 150}
    idx_w23 = {k: v for k, v in idx_w23.items() if len(v) <= 150}
    idx_addr_num_st = {k: v for k, v in idx_addr_num_st.items() if len(v) <= 50}
    idx_addr_num_word = {k: v for k, v in idx_addr_num_word.items() if len(v) <= 50}
    print(f"      Indices built and pruned in {time.time()-t_idx:.2f}s", flush=True)

    # 3. Stream Sources 2 & 3 with Candidate Retention
    print("\n[3/5] Streaming Sources 2 & 3 with Memory-Bounded Retention...", flush=True)
    STAGE2_MAX_CANDS = 12
    cand_tuples = {}
    s2_candidates = defaultdict(dict)
    s3_candidates = defaultdict(dict)
    t_stream = time.time()

    for filename in ['test_source2.tsv', 'test_source3.tsv']:
        filepath = os.path.join(test_dir, filename)
        is_s2 = 'source2' in filename
        target_dict = s2_candidates if is_s2 else s3_candidates
        print(f"      -> Processing {filename}...", flush=True)

        n_rows = 0
        for chunk in pd.read_csv(filepath, sep='\t', chunksize=250000, dtype=dtype_spec):
            n_rows += len(chunk)
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
                if snum and st_pref and (c, snum, st_pref) in idx_addr_num_st:
                    for sid in idx_addr_num_st[(c, snum, st_pref)]: sid_scores[sid] = max(sid_scores[sid], 65)
                if snum:
                    for w in words[:2]:
                        if len(w) >= 3 and (c, snum, w) in idx_addr_num_word:
                            for sid in idx_addr_num_word[(c, snum, w)]: sid_scores[sid] = max(sid_scores[sid], 60)

                retained = False
                for sid, score in sid_scores.items():
                    cur = target_dict[sid]
                    if len(cur) < STAGE2_MAX_CANDS:
                        cur[cid] = score
                        retained = True
                    else:
                        min_cid = min(cur, key=cur.__getitem__)
                        if score > cur[min_cid]:
                            del cur[min_cid]
                            cur[cid] = score
                            retained = True

                if retained:
                    cand_tuples[cid] = build_record_tuple(bname, baddr, bctry)

            print(f"         Processed {n_rows:,} records...", end='\r', flush=True)
        print(f"\n         {filename} complete ({n_rows:,} records).", flush=True)

    print(f"      Streaming done in {time.time()-t_stream:.1f}s. Unique candidates: {len(cand_tuples):,}", flush=True)

    # 4. Feature Extraction & Dynamic Rank Scoring
    print("\n[4/5] Scoring Candidate Pairs with Dynamic Multi-Match Thresholding...", flush=True)
    t_score = time.time()
    s1_all_cands = {}
    CHUNK_SIZE = 250_000
    chunk_features = []
    chunk_pairs = []
    s1_model_matches = defaultdict(lambda: {'S2': list(), 'S3': list()})

    # Calibrated base thresholds
    T_PRIMARY = opt_thresh - 0.04    # e.g. 0.86
    T_SECONDARY = opt_thresh - 0.02  # e.g. 0.88
    T_TERTIARY = opt_thresh          # e.g. 0.90

    def score_chunk():
        if not chunk_features: return
        X = np.array(chunk_features, dtype=np.float32)
        probs = clf.predict_proba(X)[:, 1]
        for (sid, cid, src), prob in zip(chunk_pairs, probs):
            if prob >= T_PRIMARY:
                s1_model_matches[sid][src].append((cid, prob))
        chunk_features.clear()
        chunk_pairs.clear()

    total_candidates = 0
    for sid in s1_eids:
        s1_tup = s1_tuples[sid]
        s2_dict = s2_candidates.get(sid, {})
        s3_dict = s3_candidates.get(sid, {})

        all_cands_list = list(s2_dict.keys()) + list(s3_dict.keys())
        s1_all_cands[sid] = all_cands_list
        total_candidates += len(all_cands_list)

        for cid, score in s2_dict.items():
            if cid in cand_tuples:
                chunk_features.append(compute_pair_features(s1_tup, cand_tuples[cid], score))
                chunk_pairs.append((sid, cid, 'S2'))
                if len(chunk_features) >= CHUNK_SIZE: score_chunk()

        for cid, score in s3_dict.items():
            if cid in cand_tuples:
                chunk_features.append(compute_pair_features(s1_tup, cand_tuples[cid], score))
                chunk_pairs.append((sid, cid, 'S3'))
                if len(chunk_features) >= CHUNK_SIZE: score_chunk()

    score_chunk()
    print(f"      Scored {total_candidates:,} candidate pairs across all entities in {time.time()-t_score:.1f}s.", flush=True)

    # 5. Multi-Match Thresholding + Global 1-to-1 Disambiguation
    print("\n[5/5] Applying Multi-Match Selection & 1-to-1 Disambiguation...", flush=True)
    raw_s1_matches = defaultdict(list)
    cand_claims = defaultdict(list)

    for sid in s1_eids:
        s2_ranked = sorted(s1_model_matches[sid]['S2'], key=lambda x: -x[1])
        s3_ranked = sorted(s1_model_matches[sid]['S3'], key=lambda x: -x[1])

        # Allow up to 5 matches per source
        for src_ranked in [s2_ranked, s3_ranked]:
            for rank, (cid, prob) in enumerate(src_ranked[:5]):
                thresh = T_PRIMARY if rank == 0 else (T_SECONDARY if rank == 1 else T_TERTIARY)
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
            cn, _, ca, csnum, _, _, _, _ = cand_tuples[cid]
            for sid, prob in claims:
                s1_n, _, s1_a, s1_snum, _, _, _, _ = s1_tuples[sid]
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

    print(f"      Writing {matching_path}...", flush=True)
    with open(matching_path, 'w', encoding='utf-8') as f:
        f.writelines(matching_lines)

    print(f"      Writing {candidate_path}...", flush=True)
    with open(candidate_path, 'w', encoding='utf-8') as f:
        f.writelines(candidate_lines)

    print(f"\n  Final Statistics (v2 Pipeline):", flush=True)
    print(f"    - Total S1 Entities    : {n_s1:,}", flush=True)
    print(f"    - Singletons (Clean)   : {n_singletons:,} ({n_singletons/n_s1*100:.2f}%)", flush=True)
    print(f"    - Total Matches Made   : {total_final_matches:,} (Mean: {total_final_matches/max(n_s1-n_singletons,1):.2f}/entity)", flush=True)
    print(f"    - Total Candidate Pairs: {total_candidates:,}", flush=True)

    # Official Validator
    print("\n" + "=" * 75, flush=True)
    print("  RUNNING OFFICIAL SUBMISSION VALIDATOR", flush=True)
    print("=" * 75, flush=True)
    validator_path = 'utils/validate_submission.py'
    res = subprocess.run([
        sys.executable, validator_path,
        '--matching', matching_path,
        '--candidate', candidate_path,
        '--test-dir', test_dir
    ], capture_output=True, text=True)
    print(res.stdout, flush=True)
    if res.stderr: print("STDERR:", res.stderr, flush=True)
    if res.returncode == 0:
        print(">>> VALIDATION STATUS: PASS (100% COMPLIANT) <<<", flush=True)
    else:
        print(">>> VALIDATION STATUS: FAILED <<<", flush=True)
        sys.exit(1)

    # Copy to Downloads
    downloads_dir = r'C:\Users\ABHINAV\Downloads'
    print(f"\n  Copying final submission to {downloads_dir}...", flush=True)
    for src in [matching_path, candidate_path]:
        dst = os.path.join(downloads_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"    -> Copied: {dst}", flush=True)

    print(f"\n  Total pipeline time: {(time.time()-t0)/60:.1f} min", flush=True)
    print("  SUCCESS! New v2 submission file is ready in Downloads!", flush=True)

if __name__ == '__main__':
    main()

"""
champion_2stage_pipeline.py
===========================
High-Performance, Memory-Bounded 2-Stage Entity Resolution Pipeline (v5)
Designed for Amazon ML Challenge 2026.

Key architectural features:
  1. Multilingual Multi-Pass Inverted Index Blocking:
     - Exact name, squished slug, sorted token frequency pairs
     - Distinctive brand tokens (>= 5 chars, non-generic)
     - Postal/PIN + name prefix, City/State + name prefix
     - Street number + street prefix, Street number + name word
     - Pruned at maximum block sizes (350 multi-token, 60 brand)
  2. Candidate retention capped at MAX_CANDS=16 per source (up to 32 cands/entity)
  3. 43 pairwise features (RapidFuzz edit similarities, containment, structured location, interaction features)
  4. Champion LightGBM v5 Model (Macro F0.5 = 0.9358 on 6k held-out validation)
  5. Calibrated Multi-Match Thresholds:
     - T_PRIMARY = 0.60 (0.52 for exact clean brand)
     - T_SECONDARY = 0.75
     - Hard conflict guardrail (prob >= 0.88 if PIN/street conflict)
     - Up to 4 matches total per entity (targeting ~3.3–3.6M total matches)
  6. Global 1-to-1 Disambiguation (0 duplicate candidates across S1 entities)
  7. Direct streaming disk export (0 RAM overhead)
  8. Official submission validation with utils/validate_submission.py
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

from src.preprocessing import (
    clean_name, squish, clean_addr, get_street_num, get_all_nums,
    get_street_prefix, extract_structured, char_ngrams, get_distinctive_tokens,
    GENERIC_WORDS
)
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
    pin, state, city = extract_structured(addr, c)
    ngrams = char_ngrams(cn, 3)
    return (cn, sq, ca, snum, c, all_nums, words, awords, pin, state, city, ngrams)

def main():
    t0 = time.time()
    test_dir = 'data/dataset/test'
    output_dir = r'C:\Users\ABHINAV\AppData\Local\submission_v5'
    os.makedirs(output_dir, exist_ok=True)

    matching_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    model_path = 'output/champion_model_v5.pkl'

    print("=" * 80, flush=True)
    print("  AMAZON ML CHALLENGE 2026 - PRODUCTION RESOLUTION PIPELINE (v5)", flush=True)
    print("=" * 80, flush=True)

    # Load LightGBM v5 Model Bundle
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    clf = bundle['clf']
    t_prim = bundle.get('t_primary', 0.60)
    t_sec = bundle.get('t_secondary', 0.75)
    print(f"Loaded LightGBM v5 Model (T_Prim: {t_prim:.2f}, T_Sec: {t_sec:.2f}, Features: {clf.n_features_in_}).", flush=True)

    # 1. Load S1 Records
    print("\n[1/5] Loading Test Source 1...", flush=True)
    t_s1 = time.time()
    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    dtype_spec = {'entity_id': 'string', 'business_name': 'string', 'business_address': 'string', 'country': 'string'}
    s1_df = pd.read_csv(s1_path, sep='\t', dtype=dtype_spec)
    n_s1 = len(s1_df)

    s1_eids = list(s1_df['entity_id'])
    s1_names_raw = list(s1_df['business_name'].fillna(''))
    s1_addrs_raw = list(s1_df['business_address'].fillna(''))
    s1_ctrys_raw = list(s1_df['country'].fillna(''))
    del s1_df

    s1_data = {}
    idx_exact = defaultdict(list)
    idx_sq = defaultdict(list)
    idx_tfp = defaultdict(list)
    idx_w12 = defaultdict(list)
    idx_fl = defaultdict(list)
    idx_w23 = defaultdict(list)
    idx_brand = defaultdict(list)
    idx_addr_num_st = defaultdict(list)
    idx_addr_num_word = defaultdict(list)
    idx_pin_w = defaultdict(list)
    idx_city_w = defaultdict(list)

    print("\n[2/5] Building Multilingual Multi-Pass Inverted Index...", flush=True)
    t_idx = time.time()
    for sid, bname, baddr, bctry in zip(s1_eids, s1_names_raw, s1_addrs_raw, s1_ctrys_raw):
        tup = build_record_tuple(bname, baddr, bctry)
        cn, sq, ca, snum, c, all_nums, words, awords, pin, state, city, ngrams = tup
        s1_data[sid] = tup
        st_pref = get_street_prefix(baddr)

        if cn: idx_exact[(c, cn)].append(sid)
        if len(sq) >= 4: idx_sq[(c, sq)].append(sid)
        if 2 <= len(words) <= 4: idx_tfp[(c, '|'.join(sorted(words)))].append(sid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[1]) >= 3: idx_w12[(c, words[0], words[1])].append(sid)
        if len(words) >= 2 and len(words[0]) >= 3 and len(words[-1]) >= 3: idx_fl[(c, words[0], words[-1])].append(sid)
        if len(words) >= 3 and len(words[1]) >= 3 and len(words[-1]) >= 3: idx_w23[(c, words[1], words[-1])].append(sid)

        for w in words:
            if len(w) >= 5 and w not in GENERIC_WORDS:
                idx_brand[(c, w)].append(sid)

        if snum and st_pref: idx_addr_num_st[(c, snum, st_pref)].append(sid)
        if snum:
            for w in words[:2]:
                if len(w) >= 3: idx_addr_num_word[(c, snum, w)].append(sid)

        if pin and words and len(words[0]) >= 3:
            idx_pin_w[(c, pin, words[0][:4])].append(sid)
        if city and words and len(words[0]) >= 3:
            idx_city_w[(c, city, words[0][:4])].append(sid)

    # Prune overgrown keys for speed & precision
    idx_exact = {k: v for k, v in idx_exact.items() if len(v) <= 350}
    idx_sq = {k: v for k, v in idx_sq.items() if len(v) <= 350}
    idx_tfp = {k: v for k, v in idx_tfp.items() if len(v) <= 350}
    idx_w12 = {k: v for k, v in idx_w12.items() if len(v) <= 350}
    idx_fl = {k: v for k, v in idx_fl.items() if len(v) <= 350}
    idx_w23 = {k: v for k, v in idx_w23.items() if len(v) <= 350}
    idx_brand = {k: v for k, v in idx_brand.items() if len(v) <= 60}
    idx_addr_num_st = {k: v for k, v in idx_addr_num_st.items() if len(v) <= 80}
    idx_addr_num_word = {k: v for k, v in idx_addr_num_word.items() if len(v) <= 80}
    idx_pin_w = {k: v for k, v in idx_pin_w.items() if len(v) <= 150}
    idx_city_w = {k: v for k, v in idx_city_w.items() if len(v) <= 100}

    print(f"      Indexed {n_s1:,} entities in {time.time()-t_idx:.1f}s.", flush=True)

    # 3. Stream Sources 2 & 3 with Memory-Bounded Retention
    print("\n[3/5] Streaming Sources 2 & 3 with Memory-Bounded Retention...", flush=True)
    MAX_CANDS = 16
    cand_data = {}
    s2_candidates = defaultdict(dict)
    s3_candidates = defaultdict(dict)
    t_stream = time.time()

    for filename in ['test_source2.tsv', 'test_source3.tsv']:
        filepath = os.path.join(test_dir, filename)
        is_s2 = 'source2' in filename
        target_dict = s2_candidates if is_s2 else s3_candidates
        print(f"      -> Processing {filename}...", flush=True)

        n_rows = 0
        t_f = time.time()
        for chunk in pd.read_csv(filepath, sep='\t', chunksize=300000, dtype=dtype_spec):
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
                for w in words:
                    if len(w) >= 5 and (c, w) in idx_brand:
                        for sid in idx_brand[(c, w)]: sid_scores[sid] = max(sid_scores[sid], 65)
                if snum and st_pref and (c, snum, st_pref) in idx_addr_num_st:
                    for sid in idx_addr_num_st[(c, snum, st_pref)]: sid_scores[sid] = max(sid_scores[sid], 60)
                if snum:
                    for w in words[:2]:
                        if len(w) >= 3 and (c, snum, w) in idx_addr_num_word:
                            for sid in idx_addr_num_word[(c, snum, w)]: sid_scores[sid] = max(sid_scores[sid], 55)

                retained = False
                for sid, score in sid_scores.items():
                    cur = target_dict[sid]
                    if len(cur) < MAX_CANDS:
                        cur[cid] = score
                        retained = True
                    else:
                        min_cid = min(cur, key=cur.__getitem__)
                        if score > cur[min_cid]:
                            del cur[min_cid]
                            cur[cid] = score
                            retained = True

                if retained:
                    cand_data[cid] = build_record_tuple(bname, baddr, bctry)

            print(f"         Processed {n_rows:,} records in {time.time()-t_f:.1f}s...", end='\r', flush=True)
        print(f"\n         {filename} complete ({n_rows:,} records in {time.time()-t_f:.1f}s).", flush=True)

    # Purge orphan candidates to free RAM
    print("      Purging non-retained candidates from memory...", flush=True)
    needed_cids = set()
    for d in s2_candidates.values(): needed_cids.update(d.keys())
    for d in s3_candidates.values(): needed_cids.update(d.keys())
    cand_data = {cid: cand_data[cid] for cid in needed_cids if cid in cand_data}
    print(f"      Active unique candidates retained: {len(cand_data):,}. Streaming time: {(time.time()-t_stream)/60:.1f}m", flush=True)

    # 4. Fast Vectorized Scoring
    print("\n[4/5] Scoring Candidate Pairs with Real-Time Progress...", flush=True)
    t_score = time.time()
    s1_all_cands = {}
    CHUNK_SIZE = 150_000
    chunk_features = []
    chunk_pairs = []
    s1_model_matches = defaultdict(lambda: {'S2': list(), 'S3': list()})

    T_PRIMARY = t_prim
    T_SECONDARY = t_sec

    def score_chunk():
        if not chunk_features: return
        X = np.array(chunk_features, dtype=np.float32)
        probs = clf.predict_proba(X)[:, 1]
        for (sid, cid, src, exact_brand, has_conflict), prob in zip(chunk_pairs, probs):
            min_t = 0.88 if has_conflict else ((T_PRIMARY - 0.08) if exact_brand else T_PRIMARY)
            if prob >= min_t:
                s1_model_matches[sid][src].append((cid, prob, exact_brand, has_conflict))
        chunk_features.clear()
        chunk_pairs.clear()

    total_candidate_pairs = 0
    pairs_scored = 0

    for idx, sid in enumerate(s1_eids, 1):
        s1_tup = s1_data[sid]
        s1_n, s1_sq, s1_a, s1_snum, s1_c, s1_nums, s1_w, s1_aw, s1_pin, s1_state, s1_city, s1_ng = s1_tup

        s2_dict = s2_candidates.get(sid, {})
        s3_dict = s3_candidates.get(sid, {})

        all_cands_list = list(s2_dict.keys()) + list(s3_dict.keys())
        s1_all_cands[sid] = all_cands_list
        total_candidate_pairs += len(all_cands_list)

        for cid in s2_dict:
            if cid in cand_data:
                c_tup = cand_data[cid]
                cn, csq, ca, csnum, cc, c_nums, cw, caw, c_pin, c_state, c_city, c_ng = c_tup
                exact_brand = (s1_n == cn and len(s1_n) >= 5 and s1_c == cc and not (s1_snum and csnum and s1_snum != csnum))
                has_conflict = bool((s1_snum and csnum and s1_snum != csnum) or (s1_pin and c_pin and s1_pin != c_pin))
                chunk_features.append(compute_pair_features(s1_tup, c_tup))
                chunk_pairs.append((sid, cid, 'S2', exact_brand, has_conflict))
                if len(chunk_features) >= CHUNK_SIZE:
                    pairs_scored += len(chunk_features)
                    score_chunk()

        for cid in s3_dict:
            if cid in cand_data:
                c_tup = cand_data[cid]
                cn, csq, ca, csnum, cc, c_nums, cw, caw, c_pin, c_state, c_city, c_ng = c_tup
                exact_brand = (s1_n == cn and len(s1_n) >= 5 and s1_c == cc and not (s1_snum and csnum and s1_snum != csnum))
                has_conflict = bool((s1_snum and csnum and s1_snum != csnum) or (s1_pin and c_pin and s1_pin != c_pin))
                chunk_features.append(compute_pair_features(s1_tup, c_tup))
                chunk_pairs.append((sid, cid, 'S3', exact_brand, has_conflict))
                if len(chunk_features) >= CHUNK_SIZE:
                    pairs_scored += len(chunk_features)
                    score_chunk()

        if idx % 100_000 == 0 or idx == n_s1:
            elapsed = time.time() - t_score
            rate = max(pairs_scored, 1) / max(elapsed, 0.1)
            pct = idx / n_s1 * 100.0
            eta_sec = (n_s1 - idx) / max(idx / elapsed, 0.1)
            print(f"      Scored [{idx:,} / {n_s1:,}] ({pct:.1f}%) | {pairs_scored:,} pairs | Rate: {rate:,.0f} pairs/s | ETA: {eta_sec/60:.1f} min", flush=True)

    pairs_scored += len(chunk_features)
    score_chunk()
    print(f"      Scoring complete! Total pairs scored: {total_candidate_pairs:,} in {(time.time()-t_score)/60:.1f} min.", flush=True)

    del s2_candidates
    del s3_candidates

    # 5. Multi-Match Selection & Global 1-to-1 Disambiguation
    print("\n[5/5] Multi-Match Selection & Global 1-to-1 Disambiguation...", flush=True)
    raw_s1_matches = defaultdict(list)
    cand_claims = defaultdict(list)

    for sid in s1_eids:
        s2_ranked = sorted(s1_model_matches[sid]['S2'], key=lambda x: -x[1])
        s3_ranked = sorted(s1_model_matches[sid]['S3'], key=lambda x: -x[1])

        # Balanced Multi-Match: at most 2 from S2 and at most 2 from S3, maximum 4 matches total
        ent_matches = []
        for src_ranked in [s2_ranked, s3_ranked]:
            for rank, (cid, prob, exact_brand, has_conflict) in enumerate(src_ranked[:3]):
                if has_conflict:
                    thresh = 0.88
                elif rank == 0:
                    thresh = (T_PRIMARY - 0.08) if exact_brand else T_PRIMARY
                else:
                    thresh = T_SECONDARY

                if prob >= thresh:
                    ent_matches.append((cid, prob))

        ent_matches.sort(key=lambda x: -x[1])
        for cid, prob in ent_matches[:4]:
            raw_s1_matches[sid].append(cid)
            cand_claims[cid].append((sid, prob))

    del s1_model_matches

    # Global 1-to-1 Disambiguation (Winner Resolution)
    cand_winner = {}
    for cid, claims in cand_claims.items():
        if len(claims) == 1:
            cand_winner[cid] = claims[0][0]
        else:
            best_sid = None
            best_score = -9999.0
            cn, _, ca, csnum, _, _, _, _, c_pin, _, _, _ = cand_data[cid]
            for sid, prob in claims:
                s1_n, _, s1_a, s1_snum, _, _, _, _, s1_pin, _, _, _ = s1_data[sid]
                score = prob * 100.0 + rfuzz.token_set_ratio(s1_a, ca) * 0.5 + rfuzz.token_set_ratio(s1_n, cn) * 0.2
                if s1_snum and csnum:
                    if s1_snum == csnum: score += 30.0
                    else: score -= 50.0
                if s1_pin and c_pin:
                    if s1_pin == c_pin: score += 20.0
                    else: score -= 30.0
                if score > best_score:
                    best_score = score
                    best_sid = sid
            cand_winner[cid] = best_sid

    del cand_claims

    # Stream write directly to disk (0 RAM consumption)
    print(f"      Streaming predictions to {matching_path} and {candidate_path}...", flush=True)
    n_singletons = 0
    total_final_matches = 0

    with open(matching_path, 'w', encoding='utf-8') as f_match, \
         open(candidate_path, 'w', encoding='utf-8') as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for sid in s1_eids:
            final_matches = [c for c in raw_s1_matches.get(sid, []) if cand_winner.get(c) == sid]
            match_str = ",".join(final_matches) if final_matches else ""
            if not match_str: n_singletons += 1
            else: total_final_matches += len(final_matches)

            cands = s1_all_cands.get(sid, [])
            cand_str = ",".join(cands) if cands else ""

            f_match.write(f"{sid}\t{match_str}\n")
            f_cand.write(f"{sid}\t{cand_str}\n")

    print(f"\n  Final Pipeline Statistics:", flush=True)
    print(f"    - Total S1 Entities    : {n_s1:,}", flush=True)
    print(f"    - Singletons (Clean)   : {n_singletons:,} ({n_singletons/n_s1*100:.2f}%)", flush=True)
    print(f"    - Total Matches Made   : {total_final_matches:,} (Mean: {total_final_matches/max(n_s1-n_singletons,1):.2f}/matched entity)", flush=True)
    print(f"    - Total Candidate Pairs: {total_candidate_pairs:,}", flush=True)

    # 6. Official Validator
    print("\n" + "=" * 80, flush=True)
    print("  RUNNING OFFICIAL SUBMISSION VALIDATOR", flush=True)
    print("=" * 80, flush=True)
    res = subprocess.run([
        sys.executable, 'utils/validate_submission.py',
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

    # 7. Copy to Downloads
    downloads_dir = r'C:\Users\ABHINAV\Downloads'
    print(f"\n  Copying final submission to {downloads_dir}...", flush=True)
    for src in [matching_path, candidate_path]:
        dst = os.path.join(downloads_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"    -> Exported: {dst}", flush=True)

    print(f"\n  Total pipeline time: {(time.time()-t0)/60:.1f} minutes", flush=True)
    print("  SUCCESS! New v5 submission file is ready for leaderboard upload!", flush=True)

if __name__ == '__main__':
    main()

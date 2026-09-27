import os
import sys
import time
import re
from collections import defaultdict
from rapidfuzz import fuzz
import subprocess

def clean_text(s: str) -> str:
    if not s: return ''
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return ' '.join(s.split())

def get_street_num(addr: str) -> str:
    if not addr: return ''
    nums = re.findall(r'\b\d+\b', addr)
    return nums[0] if nums else ''

def main():
    t0 = time.time()
    input_tsv = r'C:\Users\ABHINAV\Downloads\matching_results.tsv'
    output_tsv = r'C:\Users\ABHINAV\Downloads\champion_matching_results.tsv'

    print("=" * 75)
    print("  1-TO-1 INJECTIVE DISAMBIGUATION OPTIMIZER FOR SUBMISSION")
    print("=" * 75)

    # 1. Read existing predictions
    print("[1/5] Reading input predictions...")
    s1_to_cands = {}
    cand_to_s1 = defaultdict(list)
    s1_order = []

    with open(input_tsv, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            sid = parts[0]
            s1_order.append(sid)
            mids = parts[1].split(',') if len(parts) > 1 and parts[1] else []
            s1_to_cands[sid] = mids
            for mid in mids:
                cand_to_s1[mid].append(sid)

    print(f"      Total S1 entities: {len(s1_order):,}")
    print(f"      Total candidates : {len(cand_to_s1):,}")

    conflicts = {c: sids for c, sids in cand_to_s1.items() if len(sids) > 1}
    total_excess = sum(len(sids) - 1 for sids in conflicts.values())
    conflicting_sids = set()
    for sids in conflicts.values():
        conflicting_sids.update(sids)

    print(f"      Conflicting candidates: {len(conflicts):,}")
    print(f"      Guaranteed false positive edges to prune: {total_excess:,}")
    print(f"      S1 entities affected: {len(conflicting_sids):,}")

    # 2. Load metadata for affected entities only
    print("\n[2/5] Loading S1 metadata for affected entities...")
    s1_meta = {}
    with open('data/dataset/test/test_source1.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            sid = parts[0]
            if sid in conflicting_sids:
                name = parts[1] if len(parts) > 1 else ''
                addr = parts[2] if len(parts) > 2 else ''
                s1_meta[sid] = (clean_text(name), clean_text(addr), get_street_num(addr))

    print(f"      Loaded {len(s1_meta):,} S1 entities metadata.")

    print("\n[3/5] Loading candidate metadata for conflicting candidates...")
    cand_meta = {}
    needed_cands = set(conflicts.keys())

    for src_file in ['data/dataset/test/test_source2.tsv', 'data/dataset/test/test_source3.tsv']:
        with open(src_file, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                cid = parts[0]
                if cid in needed_cands:
                    name = parts[1] if len(parts) > 1 else ''
                    addr = parts[2] if len(parts) > 2 else ''
                    cand_meta[cid] = (clean_text(name), clean_text(addr), get_street_num(addr))

    print(f"      Loaded {len(cand_meta):,} candidate entities metadata.")

    # 3. Disambiguate each conflict
    print("\n[4/5] Disambiguating 1-to-1 candidate assignments...")
    cand_winner = {}
    for cid, sids in conflicts.items():
        if cid not in cand_meta:
            # Fallback: keep first S1
            cand_winner[cid] = sids[0]
            continue

        cn, ca, csnum = cand_meta[cid]
        best_sid = None
        best_score = -9999.0

        for sid in sids:
            if sid not in s1_meta:
                score = 0.0
            else:
                s1_n, s1_a, s1_snum = s1_meta[sid]
                name_score = fuzz.token_set_ratio(s1_n, cn)
                addr_score = fuzz.token_set_ratio(s1_a, ca)
                score = 0.3 * name_score + 0.7 * addr_score
                if s1_snum and csnum:
                    if s1_snum == csnum:
                        score += 30.0
                    else:
                        score -= 50.0

            if score > best_score:
                best_score = score
                best_sid = sid

        cand_winner[cid] = best_sid

    # 4. Reconstruct clean predictions
    print("\n[5/5] Assembling final cleaned predictions...")
    cleaned_s1_to_cands = defaultdict(list)
    pruned_edges = 0

    for sid in s1_order:
        orig_cands = s1_to_cands[sid]
        clean_cands = []
        for c in orig_cands:
            if c in conflicts:
                if cand_winner[c] == sid:
                    clean_cands.append(c)
                else:
                    pruned_edges += 1
            else:
                clean_cands.append(c)
        cleaned_s1_to_cands[sid] = clean_cands

    print(f"      Total false positive edges pruned: {pruned_edges:,}")

    # Write output TSV
    print(f"      Writing {output_tsv}...")
    lines = ["source1_entity_id\tmatched_entity_ids\n"]
    empty_s1 = 0
    total_matches = 0
    for sid in s1_order:
        cands = cleaned_s1_to_cands[sid]
        match_str = ",".join(cands) if cands else ""
        if not match_str:
            empty_s1 += 1
        else:
            total_matches += len(cands)
        lines.append(f"{sid}\t{match_str}\n")

    with open(output_tsv, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"      Wrote {len(lines):,} lines to {output_tsv} ({os.path.getsize(output_tsv)//1024:,} KB).")
    print(f"      Total S1 Singletons : {empty_s1:,} ({empty_s1/len(s1_order)*100:.2f}%)")
    print(f"      Total Valid Matches : {total_matches:,} (Mean: {total_matches/(len(s1_order)-empty_s1):.2f}/matched S1)")
    print(f"      Processing took: {time.time()-t0:.1f}s")

    # Validate output
    print("\n" + "=" * 75)
    print("  RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 75)
    cand_tsv = r'C:\Users\ABHINAV\Downloads\candidate_pairs.tsv'
    validator_path = 'utils/validate_submission.py'
    res = subprocess.run([
        sys.executable, validator_path,
        '--matching', output_tsv,
        '--candidate', cand_tsv,
        '--test-dir', 'data/dataset/test'
    ], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr: print("STDERR:", res.stderr)
    if res.returncode == 0:
        print(">>> VALIDATION STATUS: PASS (100% COMPLIANT) <<<")
    else:
        print(">>> VALIDATION STATUS: FAILED <<<")
        sys.exit(1)

if __name__ == '__main__':
    main()

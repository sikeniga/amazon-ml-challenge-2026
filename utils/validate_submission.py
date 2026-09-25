"""
validate_submission.py
======================
Validates matching_results.tsv and candidate_pairs.tsv against all
official submission rules.  Uses stdlib only — no extra dependencies.

Prints PASS (exit 0) or a numbered list of issues (exit 1).

Usage (from project root):
    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""

import argparse
import csv
import os
import sys


def load_tsv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_entity_ids(path: str) -> set[str]:
    ids: set = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids.add(row["entity_id"])
    return ids


def validate(matching_path: str, candidate_path: str, test_dir: str) -> list[str]:
    issues: list[str] = []

    # Load test entity sets
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    for p in [s1_path, s2_path, s3_path]:
        if not os.path.exists(p):
            issues.append(f"Test file not found: {p}")
            return issues

    s1_ids = load_entity_ids(s1_path)
    s2_ids = load_entity_ids(s2_path)
    s3_ids = load_entity_ids(s3_path)
    valid_cand_ids = s2_ids | s3_ids

    # ---- Validate matching_results.tsv ----
    if not os.path.exists(matching_path):
        issues.append(f"matching_results.tsv not found: {matching_path}")
        return issues

    matching_rows = load_tsv(matching_path)
    required_cols = {"source1_entity_id", "matched_entity_ids"}
    if not required_cols.issubset(set(matching_rows[0].keys())):
        issues.append(f"matching_results.tsv is missing required columns: {required_cols}")
        return issues

    seen_s1: set = set()
    for row in matching_rows:
        s1_id = row["source1_entity_id"].strip()
        matched_str = row.get("matched_entity_ids", "").strip()

        # Duplicate S1 rows
        if s1_id in seen_s1:
            issues.append(f"Duplicate source1_entity_id in matching_results.tsv: {s1_id}")
        seen_s1.add(s1_id)

        # S1 ID must exist in test set
        if s1_id not in s1_ids:
            issues.append(f"source1_entity_id not in test_source1: {s1_id}")

        if not matched_str:
            continue

        matched_list = [x.strip() for x in matched_str.split(",")]

        # Duplicate IDs within the list
        if len(matched_list) != len(set(matched_list)):
            issues.append(f"Duplicate matched IDs for {s1_id}")

        # Must reference S2/S3 only
        for mid in matched_list:
            if mid not in valid_cand_ids:
                issues.append(f"Invalid matched_entity_id {mid!r} for {s1_id} "
                              f"(not in S2 or S3 test set)")

    # Every S1 entity must appear
    missing_s1 = s1_ids - seen_s1
    if missing_s1:
        issues.append(
            f"Missing {len(missing_s1)} S1 entities in matching_results.tsv "
            f"(first 5: {sorted(missing_s1)[:5]})"
        )

    # ---- Validate candidate_pairs.tsv ----
    if not os.path.exists(candidate_path):
        issues.append(f"candidate_pairs.tsv not found: {candidate_path}")
        return issues

    candidate_rows = load_tsv(candidate_path)
    cand_required = {"source1_entity_id", "candidate_entity_ids"}
    if not cand_required.issubset(set(candidate_rows[0].keys())):
        issues.append(f"candidate_pairs.tsv is missing required columns: {cand_required}")
        return issues

    # Build candidate set for cross-check
    candidate_set: dict = {}
    seen_s1_cand: set = set()
    for row in candidate_rows:
        s1_id = row["source1_entity_id"].strip()
        cand_str = row.get("candidate_entity_ids", "").strip()

        if s1_id in seen_s1_cand:
            issues.append(f"Duplicate source1_entity_id in candidate_pairs.tsv: {s1_id}")
        seen_s1_cand.add(s1_id)

        candidate_set[s1_id] = set()
        if cand_str:
            for cid in cand_str.split(","):
                candidate_set[s1_id].add(cid.strip())

    # Every match should appear in candidates
    for row in matching_rows:
        s1_id = row["source1_entity_id"].strip()
        matched_str = row.get("matched_entity_ids", "").strip()
        if not matched_str:
            continue
        cands = candidate_set.get(s1_id, set())
        for mid in matched_str.split(","):
            mid = mid.strip()
            if mid and mid not in cands:
                issues.append(
                    f"Pipeline bug: matched ID {mid!r} for {s1_id} "
                    f"never appeared in candidate_pairs.tsv"
                )

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Directory containing test TSV files")
    args = parser.parse_args()

    issues = validate(args.matching, args.candidate, args.test_dir)

    if not issues:
        print("PASS — submission files are valid.")
        sys.exit(0)
    else:
        print(f"FAIL — {len(issues)} issue(s) found:\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()

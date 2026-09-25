"""
blocking.py
===========
Multi-strategy candidate generation (blocking) for entity resolution.

Strategies (union of all):
  1. Country filter  — same country required in all blocks below
  2. Name first-token exact match
  3. Name 3-gram prefix match
  4. Address token overlap (>=1 meaningful token in common)
  5. TF-IDF cosine on char n-grams of names (top-k per S1 entity)

The UNION of all strategies forms the candidate set written to
candidate_pairs.tsv.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_index(df: pd.DataFrame, key_col: str) -> dict:
    """Map key value -> list of entity_ids."""
    idx: dict = defaultdict(list)
    for _, row in df.iterrows():
        val = row[key_col]
        if val:
            idx[val].append(row["entity_id"])
    return idx


# ---------------------------------------------------------------------------
# Individual blocking strategies
# ---------------------------------------------------------------------------

def block_by_name_first_token(s1: pd.DataFrame, candidates: pd.DataFrame) -> set:
    """Block on exact match of first normalized name token."""
    pairs: set = set()
    cand_idx = _build_index(candidates, "name_first_token")
    for _, r in s1.iterrows():
        key = r["name_first_token"]
        if not key:
            continue
        for cid in cand_idx.get(key, []):
            pairs.add((r["entity_id"], cid))
    return pairs


def block_by_name_prefix(s1: pd.DataFrame, candidates: pd.DataFrame, length: int = 3) -> set:
    """Block on first N characters of normalized name."""
    pairs: set = set()
    cand_idx = _build_index(candidates, "name_prefix3")
    for _, r in s1.iterrows():
        key = r["name_prefix3"]
        if not key:
            continue
        for cid in cand_idx.get(key, []):
            pairs.add((r["entity_id"], cid))
    return pairs


def block_by_address_token_overlap(s1: pd.DataFrame, candidates: pd.DataFrame, min_overlap: int = 1) -> set:
    """Block on shared address tokens using an inverted index."""
    pairs: set = set()
    token_idx: dict = defaultdict(set)
    for _, row in candidates.iterrows():
        for tok in row["address_tokens"]:
            if len(tok) >= 3:          # skip very short / noisy tokens
                token_idx[tok].add(row["entity_id"])
    for _, r in s1.iterrows():
        overlap_counts: dict = defaultdict(int)
        for tok in r["address_tokens"]:
            if len(tok) >= 3:
                for cid in token_idx.get(tok, set()):
                    overlap_counts[cid] += 1
        for cid, cnt in overlap_counts.items():
            if cnt >= min_overlap:
                pairs.add((r["entity_id"], cid))
    return pairs


def block_by_tfidf_name(
    s1: pd.DataFrame,
    candidates: pd.DataFrame,
    top_k: int = 20,
    min_score: float = 0.30,
) -> set:
    """
    TF-IDF cosine similarity on char n-grams of normalized names.
    For each S1 entity, retrieve top_k candidates above min_score.
    Processed in batches to stay memory-safe.
    """
    pairs: set = set()
    if candidates.empty or s1.empty:
        return pairs

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    all_texts = pd.concat([s1["norm_name"], candidates["norm_name"]], ignore_index=True)
    vectorizer.fit(all_texts)

    s1_vecs = vectorizer.transform(s1["norm_name"])
    cand_vecs = vectorizer.transform(candidates["norm_name"])
    s1_ids = s1["entity_id"].values
    cand_ids = candidates["entity_id"].values

    batch_size = 500
    for start in range(0, len(s1_ids), batch_size):
        end = min(start + batch_size, len(s1_ids))
        sim_block = cosine_similarity(s1_vecs[start:end], cand_vecs)
        for i, row_sim in enumerate(sim_block):
            top_idx = np.argsort(row_sim)[::-1][:top_k]
            for j in top_idx:
                if row_sim[j] >= min_score:
                    pairs.add((s1_ids[start + i], cand_ids[j]))
    return pairs


# ---------------------------------------------------------------------------
# Country-aware unified blocker
# ---------------------------------------------------------------------------

def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    tfidf_top_k: int = 20,
    tfidf_min_score: float = 0.30,
) -> dict:
    """
    Run all blocking strategies per-country, then union results.

    Returns
    -------
    dict  {s1_entity_id: set_of_candidate_ids}
    """
    candidates_map: dict = defaultdict(set)
    cand_all = pd.concat([s2, s3], ignore_index=True)

    all_countries = (
        set(s1["norm_country"].dropna().unique())
        | set(cand_all["norm_country"].dropna().unique())
    )

    for country in all_countries:
        s1_c = s1[s1["norm_country"] == country]
        cand_c = cand_all[cand_all["norm_country"] == country]

        if s1_c.empty or cand_c.empty:
            continue

        print(f"  Blocking country={country!r}: {len(s1_c)} S1 records, {len(cand_c)} candidates")

        for pair_set in [
            block_by_name_first_token(s1_c, cand_c),
            block_by_name_prefix(s1_c, cand_c, length=3),
            block_by_address_token_overlap(s1_c, cand_c, min_overlap=1),
            block_by_tfidf_name(s1_c, cand_c, top_k=tfidf_top_k, min_score=tfidf_min_score),
        ]:
            for s1_id, cand_id in pair_set:
                candidates_map[s1_id].add(cand_id)

    # Guarantee every S1 entity appears (singletons -> empty set)
    for s1_id in s1["entity_id"]:
        if s1_id not in candidates_map:
            candidates_map[s1_id] = set()

    total = sum(len(v) for v in candidates_map.values())
    print(f"Total candidate pairs generated: {total:,}")
    return dict(candidates_map)


def save_candidate_pairs(candidates_map: dict, path: str) -> None:
    rows = [
        {
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(sorted(cand_ids)),
        }
        for s1_id, cand_ids in sorted(candidates_map.items())
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    print(f"Saved candidate pairs -> {path}")

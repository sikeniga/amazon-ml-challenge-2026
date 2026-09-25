"""
features.py
===========
Build the full pairwise feature matrix from candidate pairs.

Combines:
  - Name similarity features   (from name_features.py)
  - Address similarity features (from address_features.py)
  - Country match flag
  - Bulk TF-IDF cosines for names and addresses (computed here in batches)

This is the single entry point called by both train.py and inference.py.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from name_features import name_features
from address_features import address_features


# ---------------------------------------------------------------------------
# Bulk paired TF-IDF cosine  (NOT full matrix — row-wise dot product)
# ---------------------------------------------------------------------------

def _build_tfidf_lookup(
    texts_a: list,
    texts_b: list,
    ngram_range: tuple = (2, 4),
    analyzer: str = "char_wb",
) -> list:
    """
    Given two equal-length aligned lists of strings,
    return a list of cosine similarity scores (one per pair).

    This is O(N) in pairs, not O(N^2), because we compute dot-products
    row-by-row rather than building the full similarity matrix.
    """
    if not texts_a:
        return []

    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=1)
    vectorizer.fit(list(texts_a) + list(texts_b))
    vecs_a = vectorizer.transform(texts_a)
    vecs_b = vectorizer.transform(texts_b)

    cosines: list = []
    batch = 1000
    for i in range(0, len(texts_a), batch):
        a_sl = vecs_a[i: i + batch]
        b_sl = vecs_b[i: i + batch]
        dot = np.array(a_sl.multiply(b_sl).sum(axis=1)).flatten()
        norm_a = np.sqrt(np.array(a_sl.power(2).sum(axis=1)).flatten())
        norm_b = np.sqrt(np.array(b_sl.power(2).sum(axis=1)).flatten())
        denom = norm_a * norm_b
        cos = np.where(denom > 0, dot / denom, 0.0)
        cosines.extend(cos.tolist())
    return cosines


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    pairs: list,          # list of (s1_entity_id, candidate_entity_id)
    s1_df: pd.DataFrame,
    cand_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a feature DataFrame for the given candidate pairs.

    Parameters
    ----------
    pairs    : list of (source1_entity_id, candidate_entity_id) tuples
    s1_df    : preprocessed Source 1 DataFrame
    cand_df  : preprocessed combined Source2+Source3 DataFrame

    Returns
    -------
    pd.DataFrame with columns:
        source1_entity_id, candidate_entity_id, country_match,
        name_*, address_*
    """
    if not pairs:
        return pd.DataFrame()

    s1_idx = s1_df.set_index("entity_id")
    cand_idx = cand_df.set_index("entity_id")

    s1_ids = [p[0] for p in pairs]
    cand_ids = [p[1] for p in pairs]

    # Collect raw texts for bulk TF-IDF
    s1_names = [
        s1_idx.loc[i, "norm_name"] if i in s1_idx.index else "" for i in s1_ids
    ]
    cand_names = [
        cand_idx.loc[i, "norm_name"] if i in cand_idx.index else "" for i in cand_ids
    ]
    s1_addrs = [
        s1_idx.loc[i, "norm_address"] if i in s1_idx.index else "" for i in s1_ids
    ]
    cand_addrs = [
        cand_idx.loc[i, "norm_address"] if i in cand_idx.index else "" for i in cand_ids
    ]

    print(f"  Computing TF-IDF cosines for {len(pairs):,} pairs (names)...")
    name_cosines = _build_tfidf_lookup(s1_names, cand_names)

    print(f"  Computing TF-IDF cosines for {len(pairs):,} pairs (addresses)...")
    addr_cosines = _build_tfidf_lookup(s1_addrs, cand_addrs)

    rows = []
    for idx, (s1_id, cand_id) in enumerate(pairs):
        if s1_id not in s1_idx.index or cand_id not in cand_idx.index:
            continue

        r1 = s1_idx.loc[s1_id]
        rc = cand_idx.loc[cand_id]

        n_feats = name_features(
            r1["norm_name"],
            rc["norm_name"],
            list(r1["name_tokens"]),
            list(rc["name_tokens"]),
            tfidf_cosine=name_cosines[idx],
        )
        a_feats = address_features(
            r1["norm_address"],
            rc["norm_address"],
            list(r1["address_tokens"]),
            list(rc["address_tokens"]),
            tfidf_cosine=addr_cosines[idx],
        )

        row = {
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "country_match": int(r1["norm_country"] == rc["norm_country"]),
        }
        row.update(n_feats)
        row.update(a_feats)
        rows.append(row)

    return pd.DataFrame(rows)

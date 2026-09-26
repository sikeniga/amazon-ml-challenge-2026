"""
ultra_submission.py
===================
State-of-the-art Entity Resolution Pipeline for Amazon ML Challenge 2026.

Strategy to go from 0.395 â†’ ~0.98 F0.5:

PHASE 1 â€” Training (on train split):
  - Multi-key blocking with high recall (character n-grams, token sets, phonetics)
  - Rich pairwise features (19 similarity signals)
  - LightGBM classifier with F0.5-optimal threshold tuning

PHASE 2 â€” Test Inference:
  - Same blocking on test data (streaming for S2/S3)
  - Apply trained LightGBM to score each candidate pair
  - Output compact, high-precision matching_results.tsv + candidate_pairs.tsv
"""

import os
import re
import sys
import time
import pickle
import unicodedata
import shutil
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import lightgbm as lgb

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. TEXT NORMALIZATION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

LEGAL_SUFFIXES_RE = re.compile(
    r'\b(incorporated|corporation|company|limited|liability|partnership|'
    r'pvt|private|public|llc|pllc|llp|inc|corp|ltd|lp|co|pc|plc|'
    r'sarl|sas|sasu|sa|eurl|snc|sci|gie|bv|nv|ag|gmbh|kft|sl|srl|'
    r'spa|oy|ab|aps|doo|sro|ooo)\b'
)

DOMAIN_EXT_RE = re.compile(r'\b(com|org|net|in|fr|io|co|biz|info|gov|edu|uk|de|au|us)\b')

COUNTRY_MAP = {
    'usa': 'us', 'united states': 'us', 'united states of america': 'us',
    'america': 'us', 'u.s.': 'us', 'u.s.a.': 'us',
    'india': 'in', 'bharat': 'in', 'ind': 'in',
    'france': 'fr', 'french republic': 'fr', 'republique francaise': 'fr',
    'uk': 'gb', 'united kingdom': 'gb', 'england': 'gb', 'britain': 'gb',
    'germany': 'de', 'deutschland': 'de',
    'canada': 'ca', 'australia': 'au', 'japan': 'jp', 'china': 'cn',
    'brazil': 'br', 'mexico': 'mx', 'spain': 'es', 'italy': 'it',
}


def _unicode_clean(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return s.lower()


def normalize_name(name) -> str:
    s = _unicode_clean(name)
    if not s:
        return ""
    # DBA / trading as extraction
    for sep in [' d/b/a ', ' dba ', ' t/a ', ' ta ', ' trading as ', ' doing business as ']:
        if sep in s:
            s = s.split(sep)[-1]
            break
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = LEGAL_SUFFIXES_RE.sub(' ', s)
    return ' '.join(s.split())


def squish_key(norm: str) -> str:
    s = DOMAIN_EXT_RE.sub('', norm)
    return ''.join(s.split())


def normalize_country(country) -> str:
    if not isinstance(country, str) or not country.strip():
        return 'unk'
    c = country.lower().strip()
    return COUNTRY_MAP.get(c, c[:5])  # truncate very long unknown countries


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. BLOCKING INDEX (vectorized for speed)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_blocking_index_from_df(df: pd.DataFrame) -> tuple:
    """
    Build multi-key inverted index from a DataFrame.
    Returns: (index dict, lookup dict of eidâ†’(name,addr,country))
    """
    index = defaultdict(list)
    lookup = {}

    eids = df['entity_id'].astype(str).values
    names = df['business_name'].fillna('').astype(str).values
    addrs = df['business_address'].fillna('').astype(str).values
    ctrys = df['country'].fillna('').astype(str).values

    for eid, raw_name, raw_addr, raw_ctry in zip(eids, names, addrs, ctrys):
        norm = normalize_name(raw_name)
        ctry = normalize_country(raw_ctry)
        sq = squish_key(norm)
        tokens = norm.split()
        sig = [t for t in tokens if len(t) >= 3]

        lookup[eid] = (raw_name, raw_addr, raw_ctry)

        # Key 1: exact normalized name
        if norm:
            index[('EN', ctry, norm)].append(eid)

        # Key 2: squish (slug)
        if len(sq) >= 5:
            index[('SQ', ctry, sq)].append(eid)

        # Key 3: first significant token
        if tokens and len(tokens[0]) >= 3:
            index[('FT', ctry, tokens[0])].append(eid)

        # Key 4: first + last significant token pair (sorted â†’ order-invariant)
        if len(sig) >= 2:
            pair = tuple(sorted([sig[0], sig[-1]]))
            index[('FL', ctry, pair[0], pair[1])].append(eid)

        # Key 5: 3-char prefix (fuzzy match on prefix errors)
        if len(norm) >= 4:
            index[('P3', ctry, norm[:3])].append(eid)

        # Key 6: bigram set fingerprint (first 3 bigrams of norm name)
        if len(norm) >= 4:
            bigrams = [norm[i:i+2] for i in range(min(len(norm)-1, 8))]
            for bg in bigrams[:3]:
                index[('BG', ctry, bg)].append(eid)

    return dict(index), lookup


def update_index_from_chunk(chunk: pd.DataFrame, index: defaultdict, lookup: dict) -> None:
    """Add records from a DataFrame chunk into existing index and lookup."""
    eids = chunk['entity_id'].astype(str).values
    names = chunk['business_name'].fillna('').astype(str).values
    addrs = chunk['business_address'].fillna('').astype(str).values
    ctrys = chunk['country'].fillna('').astype(str).values

    for eid, raw_name, raw_addr, raw_ctry in zip(eids, names, addrs, ctrys):
        norm = normalize_name(raw_name)
        ctry = normalize_country(raw_ctry)
        sq = squish_key(norm)
        tokens = norm.split()
        sig = [t for t in tokens if len(t) >= 3]

        lookup[eid] = (raw_name, raw_addr, raw_ctry)

        if norm:
            index[('EN', ctry, norm)].append(eid)
        if len(sq) >= 5:
            index[('SQ', ctry, sq)].append(eid)
        if tokens and len(tokens[0]) >= 3:
            index[('FT', ctry, tokens[0])].append(eid)
        if len(sig) >= 2:
            pair = tuple(sorted([sig[0], sig[-1]]))
            index[('FL', ctry, pair[0], pair[1])].append(eid)
        if len(norm) >= 4:
            index[('P3', ctry, norm[:3])].append(eid)
        if len(norm) >= 4:
            bigrams = [norm[i:i+2] for i in range(min(len(norm)-1, 8))]
            for bg in bigrams[:3]:
                index[('BG', ctry, bg)].append(eid)


def lookup_candidates_for_entity(
    eid: str, norm: str, ctry_norm: str, index: dict,
    max_per_key: int = 50
) -> set:
    """Return all candidate entity IDs for one S1 entity."""
    sq = squish_key(norm)
    tokens = norm.split()
    sig = [t for t in tokens if len(t) >= 3]
    hits = set()

    # Key 1
    if norm:
        for cid in index.get(('EN', ctry_norm, norm), [])[:max_per_key]:
            if cid != eid: hits.add(cid)

    # Key 2
    if len(sq) >= 5:
        for cid in index.get(('SQ', ctry_norm, sq), [])[:max_per_key]:
            if cid != eid: hits.add(cid)

    # Key 3
    if tokens and len(tokens[0]) >= 3:
        for cid in index.get(('FT', ctry_norm, tokens[0]), [])[:max_per_key]:
            if cid != eid: hits.add(cid)

    # Key 4
    if len(sig) >= 2:
        pair = tuple(sorted([sig[0], sig[-1]]))
        for cid in index.get(('FL', ctry_norm, pair[0], pair[1]), [])[:max_per_key]:
            if cid != eid: hits.add(cid)

    # Key 5
    if len(norm) >= 4:
        for cid in index.get(('P3', ctry_norm, norm[:3]), [])[:max_per_key]:
            if cid != eid: hits.add(cid)

    # Key 6: bigrams (limit since this can be noisy)
    if len(norm) >= 4:
        bigrams = [norm[i:i+2] for i in range(min(len(norm)-1, 6))]
        for bg in bigrams[:2]:  # Only use first 2 bigrams to reduce noise
            for cid in index.get(('BG', ctry_norm, bg), [])[:15]:
                if cid != eid: hits.add(cid)

    return hits


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. SIMILARITY FEATURES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    return inter / (len(set_a) + len(set_b) - inter)


def _edit_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _token_sort_sim(a: str, b: str) -> float:
    return _edit_sim(' '.join(sorted(a.split())), ' '.join(sorted(b.split())))


def _token_set_sim(a: str, b: str) -> float:
    set_a, set_b = set(a.split()), set(b.split())
    inter = set_a & set_b
    i_str = ' '.join(sorted(inter))
    a_diff = ' '.join(sorted(set_a - inter))
    b_diff = ' '.join(sorted(set_b - inter))
    ia = (i_str + ' ' + a_diff).strip()
    ib = (i_str + ' ' + b_diff).strip()
    return max(
        _edit_sim(i_str, ia) if i_str else 0.0,
        _edit_sim(i_str, ib) if i_str else 0.0,
        _edit_sim(ia, ib),
    )


def char_ngram_jaccard(a: str, b: str, n: int) -> float:
    if len(a) < n or len(b) < n:
        return float(a == b)
    set_a = set(a[i:i+n] for i in range(len(a)-n+1))
    set_b = set(b[i:i+n] for i in range(len(b)-n+1))
    return _jaccard(set_a, set_b)


def compute_features(
    name1, name2, addr1, addr2, ctry1, ctry2
) -> dict:
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)

    # Normalize addresses
    a1 = _unicode_clean(addr1)
    a2 = _unicode_clean(addr2)
    a1 = re.sub(r'[^a-z0-9\s]', ' ', a1)
    a2 = re.sub(r'[^a-z0-9\s]', ' ', a2)
    a1 = ' '.join(a1.split())
    a2 = ' '.join(a2.split())

    c1 = normalize_country(ctry1)
    c2 = normalize_country(ctry2)

    t1 = set(n1.split())
    t2 = set(n2.split())
    at1 = set(t for t in a1.split() if len(t) >= 3)
    at2 = set(t for t in a2.split() if len(t) >= 3)

    sq1 = squish_key(n1)
    sq2 = squish_key(n2)

    return {
        # Country
        'country_match': int(c1 == c2),
        # Name exact
        'name_exact': int(n1 == n2),
        'name_squish_exact': int(sq1 == sq2 and len(sq1) >= 4),
        # Name string sims
        'name_edit_sim': _edit_sim(n1, n2),
        'name_token_sort': _token_sort_sim(n1, n2),
        'name_token_set': _token_set_sim(n1, n2),
        # N-gram Jaccard
        'name_jaccard_token': _jaccard(t1, t2),
        'name_jaccard_bigram': char_ngram_jaccard(n1, n2, 2),
        'name_jaccard_trigram': char_ngram_jaccard(n1, n2, 3),
        # Token stats
        'name_common_tokens': len(t1 & t2),
        'name_len_diff': abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1),
        'name_first_token_match': int(
            bool(n1.split()) and bool(n2.split()) and n1.split()[0] == n2.split()[0]
        ),
        # Address
        'addr_exact': int(bool(a1) and a1 == a2),
        'addr_edit_sim': _edit_sim(a1[:80], a2[:80]),  # cap length for speed
        'addr_jaccard_token': _jaccard(at1, at2),
        'addr_common_tokens': len(at1 & at2),
        # Combined
        'both_exact': int(n1 == n2 and a1 == a2 and bool(n1)),
        'max_name_sim': max(_edit_sim(n1, n2), _token_sort_sim(n1, n2)),
    }


FEATURE_COLS = [
    'country_match',
    'name_exact', 'name_squish_exact',
    'name_edit_sim', 'name_token_sort', 'name_token_set',
    'name_jaccard_token', 'name_jaccard_bigram', 'name_jaccard_trigram',
    'name_common_tokens', 'name_len_diff', 'name_first_token_match',
    'addr_exact', 'addr_edit_sim', 'addr_jaccard_token', 'addr_common_tokens',
    'both_exact', 'max_name_sim',
]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. METRICS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def macro_f05(preds: dict, gt: dict) -> float:
    """Entity-level macro F0.5 (beta=0.5 â†’ betaÂ²=0.25)."""
    beta2 = 0.25
    scores = []
    all_s1 = set(gt.keys()) | set(preds.keys())
    for s1_id in all_s1:
        true_set = gt.get(s1_id, set())
        pred_set = preds.get(s1_id, set())
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        num = (1 + beta2) * prec * rec
        den = beta2 * prec + rec
        f05 = num / den if den > 0 else 0.0
        scores.append(f05)
    return float(np.mean(scores)) if scores else 0.0


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. TRAINING
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_model(train_dir: str, output_dir: str) -> dict:
    print("\n" + "="*65)
    print("PHASE 1: TRAINING LightGBM Entity Matcher")
    print("="*65)
    t_start = time.time()

    dtype_spec = {'entity_id': 'string', 'business_name': 'string',
                  'business_address': 'string', 'country': 'string'}

    # â”€â”€ 1. Load training data â”€â”€
    print("\n[1/5] Loading training data...")
    s1 = pd.read_csv(os.path.join(train_dir, 'train_source1.tsv'), sep='\t', dtype=dtype_spec)
    gt_df = pd.read_csv(os.path.join(train_dir, 'train_ground_truth.tsv'), sep='\t')
    print(f"  S1 rows: {len(s1):,}  GT rows: {len(gt_df):,}")

    # Parse ground truth
    gt: dict = defaultdict(set)
    for _, row in gt_df.iterrows():
        s1_id = str(row['source1_entity_id'])
        raw = str(row.get('matched_entity_ids', '') or '')
        for cid in raw.split(','):
            cid = cid.strip()
            if cid:
                gt[s1_id].add(cid)
    print(f"  GT covers {len(gt):,} S1 entities, {sum(len(v) for v in gt.values()):,} total matches")

    # â”€â”€ 2. Build blocking index from S2 + S3 â”€â”€
    print("\n[2/5] Building blocking index from S2 + S3...")
    cand_index = defaultdict(list)
    cand_lookup = {}

    for fname in ['train_source2.tsv', 'train_source3.tsv']:
        fpath = os.path.join(train_dir, fname)
        print(f"  â†’ {fname}...")
        for chunk in pd.read_csv(fpath, sep='\t', dtype=dtype_spec, chunksize=300_000):
            update_index_from_chunk(chunk, cand_index, cand_lookup)
            print(f"    chunk processed, index size: {len(cand_index):,}", end='\r')
    print(f"\n  Candidate records: {len(cand_lookup):,}  Index buckets: {len(cand_index):,}")

    cand_index = dict(cand_index)

    # Build S1 lookup too
    s1_lookup = {}
    for _, row in s1.iterrows():
        eid = str(row['entity_id'])
        s1_lookup[eid] = (
            str(row.get('business_name', '') or ''),
            str(row.get('business_address', '') or ''),
            str(row.get('country', '') or ''),
        )

    # â”€â”€ 3. Generate candidate pairs â”€â”€
    print("\n[3/5] Generating candidate pairs + adding guaranteed positives...")
    all_pairs = set()

    for _, row in s1.iterrows():
        eid = str(row['entity_id'])
        raw_name = str(row.get('business_name', '') or '')
        raw_ctry = str(row.get('country', '') or '')

        norm = normalize_name(raw_name)
        ctry_norm = normalize_country(raw_ctry)

        hits = lookup_candidates_for_entity(eid, norm, ctry_norm, cand_index)

        # Always include true positives so we have training signal
        for cid in gt.get(eid, set()):
            hits.add(cid)

        for cid in hits:
            all_pairs.add((eid, cid))

    all_pairs = list(all_pairs)
    print(f"  Total candidate pairs: {len(all_pairs):,}")

    # â”€â”€ 4. Feature computation â”€â”€
    print("\n[4/5] Computing pairwise features...")
    all_lookup = {**s1_lookup, **cand_lookup}

    rows = []
    for s1_id, cid in all_pairs:
        r1 = all_lookup.get(s1_id)
        r2 = all_lookup.get(cid)
        if r1 is None or r2 is None:
            continue
        feats = compute_features(r1[0], r2[0], r1[1], r2[1], r1[2], r2[2])
        feats['source1_entity_id'] = s1_id
        feats['candidate_entity_id'] = cid
        feats['label'] = 1 if cid in gt.get(s1_id, set()) else 0
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    pos = int(feat_df['label'].sum())
    neg = len(feat_df) - pos
    print(f"  Positives: {pos:,}   Negatives: {neg:,}   Ratio: 1:{neg//max(pos,1)}")

    if pos == 0:
        print("ERROR: No positive training examples! Check data paths and ground truth format.")
        return None

    # â”€â”€ 5. Train LightGBM â”€â”€
    print("\n[5/5] Training LightGBM with early stopping...")

    # Val split by S1 entity (15% for threshold tuning)
    s1_ids_unique = feat_df['source1_entity_id'].unique()
    rng = np.random.default_rng(42)
    n_val = max(1, int(len(s1_ids_unique) * 0.15))
    val_s1 = set(rng.choice(s1_ids_unique, size=n_val, replace=False).tolist())
    is_val = feat_df['source1_entity_id'].isin(val_s1)
    train_df = feat_df[~is_val].reset_index(drop=True)
    val_df = feat_df[is_val].reset_index(drop=True)
    print(f"  Train rows: {len(train_df):,}   Val rows: {len(val_df):,}")

    X_tr = train_df[FEATURE_COLS].values.astype(np.float32)
    y_tr = train_df['label'].values
    X_val_arr = val_df[FEATURE_COLS].values.astype(np.float32)
    y_val_arr = val_df['label'].values

    scale = neg / max(pos, 1)
    clf = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.5,
        n_estimators=2000,
        scale_pos_weight=scale,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_val_arr, y_val_arr)],
        callbacks=[
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(200),
        ],
    )
    print(f"  Best iteration: {clf.best_iteration_}")

    # Feature importance
    fi = sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1])
    print("\n  Top feature importances:")
    for fname, fimp in fi[:10]:
        bar = 'â–ˆ' * int(fimp / max(fi[0][1], 1) * 30)
        print(f"    {fname:35s} {bar}")

    # Threshold tuning for F0.5
    print("\n  Tuning decision threshold on validation set (maximizing F0.5)...")
    proba_val = clf.predict_proba(X_val_arr)[:, 1]
    val_gt = defaultdict(set)
    for _, r in val_df.iterrows():
        if r['label'] == 1:
            val_gt[r['source1_entity_id']].add(r['candidate_entity_id'])

    best_thresh, best_f05 = 0.5, -1.0
    for thresh in np.arange(0.15, 0.90, 0.05):
        preds_map: dict = {}
        for prob, sid, cid in zip(proba_val, val_df['source1_entity_id'], val_df['candidate_entity_id']):
            if sid not in preds_map:
                preds_map[sid] = set()
            if prob >= thresh:
                preds_map[sid].add(cid)
        score = macro_f05(preds_map, dict(val_gt))
        marker = " â† best" if score > best_f05 else ""
        print(f"    thresh={thresh:.2f}  F0.5={score:.4f}{marker}")
        if score > best_f05:
            best_f05 = score
            best_thresh = float(thresh)

    print(f"\n  â˜… Best threshold: {best_thresh:.2f}  (Val F0.5={best_f05:.4f})")
    print(f"  Training time: {(time.time()-t_start)/60:.1f} min")

    model_data = {'clf': clf, 'threshold': best_thresh, 'val_f05': best_f05}
    model_path = os.path.join(output_dir, 'ultra_model.pkl')
    os.makedirs(output_dir, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  Model saved â†’ {model_path}")

    return model_data


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. TEST INFERENCE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_inference(test_dir: str, output_dir: str, model_data: dict) -> None:
    print("\n" + "="*65)
    print("PHASE 2: TEST INFERENCE")
    print("="*65)
    t_start = time.time()

    clf = model_data['clf']
    threshold = model_data['threshold']
    print(f"\n  Threshold: {threshold:.2f}   (Val F0.5={model_data.get('val_f05', '?'):.4f})")

    dtype_spec = {'entity_id': 'string', 'business_name': 'string',
                  'business_address': 'string', 'country': 'string'}

    # Load test S1
    print("\n[1/4] Loading Test Source 1...")
    s1 = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep='\t', dtype=dtype_spec)
    n_s1 = len(s1)
    print(f"  Loaded {n_s1:,} entities")

    s1_lookup = {}
    s1_queries = []
    for _, row in s1.iterrows():
        eid = str(row['entity_id'])
        name = str(row.get('business_name', '') or '')
        addr = str(row.get('business_address', '') or '')
        ctry = str(row.get('country', '') or '')
        s1_lookup[eid] = (name, addr, ctry)
        norm = normalize_name(name)
        ctry_norm = normalize_country(ctry)
        s1_queries.append((eid, norm, ctry_norm))

    # Stream S2 + S3 into blocking index
    print("\n[2/4] Streaming Test S2 + S3 into blocking index...")
    cand_index = defaultdict(list)
    cand_lookup = {}
    total_cands = 0

    for fname in ['test_source2.tsv', 'test_source3.tsv']:
        fpath = os.path.join(test_dir, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found")
            continue
        print(f"  â†’ {fname}...")
        for chunk in pd.read_csv(fpath, sep='\t', dtype=dtype_spec, chunksize=500_000):
            update_index_from_chunk(chunk, cand_index, cand_lookup)
            total_cands += len(chunk)
            print(f"    Processed {total_cands:,} candidates...", end='\r')

    print(f"\n  Total candidate records: {total_cands:,}   Index buckets: {len(cand_index):,}")
    cand_index = dict(cand_index)

    # Score pairs in batches
    print("\n[3/4] Blocking + Scoring candidate pairs...")
    BATCH_SIZE = 10_000
    batch_sids, batch_cids, batch_rows = [], [], []

    all_results = {eid: set() for eid, _, _ in s1_queries}
    all_candidates = {eid: set() for eid, _, _ in s1_queries}
    total_pairs = 0

    def flush_batch():
        if not batch_rows:
            return
        X = np.array([[r[c] for c in FEATURE_COLS] for r in batch_rows], dtype=np.float32)
        proba = clf.predict_proba(X)[:, 1]
        for prob, sid, cid in zip(proba, batch_sids, batch_cids):
            all_candidates[sid].add(cid)
            if prob >= threshold:
                all_results[sid].add(cid)
        batch_sids.clear()
        batch_cids.clear()
        batch_rows.clear()

    for i, (sid, norm, ctry_norm) in enumerate(s1_queries):
        hits = lookup_candidates_for_entity(sid, norm, ctry_norm, cand_index)
        total_pairs += len(hits)

        s1_name, s1_addr, s1_ctry = s1_lookup[sid]
        for cid in hits:
            if cid not in cand_lookup:
                continue
            c_name, c_addr, c_ctry = cand_lookup[cid]
            feats = compute_features(s1_name, c_name, s1_addr, c_addr, s1_ctry, c_ctry)
            batch_sids.append(sid)
            batch_cids.append(cid)
            batch_rows.append(feats)

            if len(batch_rows) >= BATCH_SIZE:
                flush_batch()

        if (i + 1) % 20000 == 0:
            flush_batch()
            matched_so_far = sum(1 for v in all_results.values() if v)
            print(f"  [{i+1:,}/{n_s1:,}]  pairs={total_pairs:,}  matched={matched_so_far:,}")

    flush_batch()

    matched_count = sum(1 for v in all_results.values() if v)
    total_matches = sum(len(v) for v in all_results.values())
    avg_cands = sum(len(v) for v in all_candidates.values()) / max(n_s1, 1)

    print(f"\n  â”€â”€â”€ Results â”€â”€â”€")
    print(f"  Total pairs scored    : {total_pairs:,}")
    print(f"  S1 entities matched   : {matched_count:,} / {n_s1:,} ({matched_count/n_s1*100:.1f}%)")
    print(f"  Total matched pairs   : {total_matches:,}")
    print(f"  Avg candidates/S1     : {avg_cands:.2f}")

    # Write TSVs
    print("\n[4/4] Writing submission TSVs...")
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')

    with open(matching_path, 'w', encoding='utf-8') as mf, \
         open(candidate_path, 'w', encoding='utf-8') as cf:
        mf.write('source1_entity_id\tmatched_entity_ids\n')
        cf.write('source1_entity_id\tcandidate_entity_ids\n')
        for _, row in s1.iterrows():
            sid = str(row['entity_id'])
            matches = sorted(all_results.get(sid, set()))
            cands = sorted(all_candidates.get(sid, set()))
            mf.write(f"{sid}\t{','.join(matches)}\n")
            cf.write(f"{sid}\t{','.join(cands)}\n")

    print(f"  â†’ matching_results.tsv  ({os.path.getsize(matching_path)//1024:,} KB)")
    print(f"  â†’ candidate_pairs.tsv   ({os.path.getsize(candidate_path)//1024:,} KB)")

    # Copy to Downloads
    for src, dst in [
        (matching_path, r'C:\Users\HP\Downloads\matching_results.tsv'),
        (candidate_path, r'C:\Users\HP\Downloads\candidate_pairs.tsv'),
    ]:
        try:
            shutil.copy2(src, dst)
            print(f"  â†’ Copied to Downloads: {dst}")
        except Exception as e:
            print(f"  WARNING: {e}")

    print(f"\n  Inference time: {(time.time()-t_start)/60:.1f} min")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7. VALIDATE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def validate(output_dir: str, test_dir: str) -> None:
    import subprocess
    validator = 'utils/validate_submission.py'
    if not os.path.exists(validator):
        print("  Validator not found â€” skipping.")
        return
    matching_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    cmd = [sys.executable, validator,
           '--matching', matching_path,
           '--candidate', candidate_path,
           '--test-dir', test_dir]
    print("  Running:", ' '.join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(res.stderr)
    if res.returncode == 0:
        print(">>> VALIDATION: PASS âœ“ Ready for submission! <<<")
    else:
        print(f">>> VALIDATION: FAILED (exit {res.returncode}) <<<")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 8. MAIN
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain even if cached model exists')
    parser.add_argument('--train-only', action='store_true',
                        help='Only run training, skip inference')
    args = parser.parse_args()

    t0 = time.time()
    train_dir = 'data/dataset/train'
    test_dir  = 'data/dataset/test'
    output_dir = 'output'

    print("=" * 65)
    print("  AMAZON ML CHALLENGE 2026 â€” ULTRA SUBMISSION ENGINE v2")
    print("=" * 65)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'ultra_model.pkl')

    if not args.retrain and os.path.exists(model_path):
        print(f"\n  Found cached model: {model_path}")
        print("  Loading (use --retrain to force retraining)...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print(f"  Threshold={model_data['threshold']:.2f}  "
              f"Val F0.5={model_data.get('val_f05', '?')}")
    else:
        model_data = train_model(train_dir, output_dir)
        if model_data is None:
            print("TRAINING FAILED â€” aborting.")
            sys.exit(1)

    if not args.train_only:
        run_inference(test_dir, output_dir, model_data)

        print("\n" + "="*65)
        print("VALIDATION")
        print("="*65)
        validate(output_dir, test_dir)

    elapsed = time.time() - t0
    print(f"\n  Total wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("\n  âœ“ DONE â€” Submit these files:")
    print(f"    â€¢ output/matching_results.tsv")
    print(f"    â€¢ output/candidate_pairs.tsv")
    print(f"    (Also copied to Downloads/)")


if __name__ == '__main__':
    main()


"""
features.py
===========
High-throughput, vectorized feature engineering for Multilingual Entity Matching.

Key features:
  1. Safe missing-address handling (avoids false conflicts on NaN address)
  2. Multi-metric name similarities (Jaro-Winkler, Token Sort, Token Set, Squish)
  3. Street number matching & numerical disjointness detection
  4. Cross-attribute interactions (High Name + Missing Addr, High Name + High Addr)
  5. Country exact matching
  6. Pure entity-intrinsic features (NO priority score bias/leak)
"""

import re
from rapidfuzz.distance import JaroWinkler
import rapidfuzz.fuzz as rfuzz

FEATURE_NAMES = [
    # 1. NAME SIMILARITIES (0-14)
    'jw_n',
    'ts_n',
    'tset_n',
    'ratio_n',
    'pr_n',
    'exact_n',
    'name_exact_clean',
    'exact_sq',
    'jw_sq',
    'name_starts_with',
    'name_contains',
    'name_jaccard',
    'name_len_diff',
    'first_tok_jw',
    'last_tok_jw',

    # 2. ADDRESS SIMILARITIES & MISSING FLAGS (15-24)
    'both_have_addr',
    'addr_either_missing',
    'jw_a',
    'ts_a',
    'tset_a',
    'exact_a',
    'addr_jaccard',
    'addr_containment',
    'addr_shared_tokens',
    'city_or_state_match',

    # 3. NUMBERS & STREET NUMBERS (22-25)
    'snum_match',
    'snum_conflict',
    'shared_nums_count',
    'disjoint_nums',

    # 4. CROSS INTERACTIONS (26-29)
    'cross_prod',
    'high_name_high_addr',
    'high_name_missing_addr',
    'high_name_addr_conflict',

    # 5. COUNTRY MATCH (30)
    'country_match'
]

def compute_pair_features(s1_tuple, cand_tuple, prio: float = 0.0) -> list:
    """
    Extract 31 high-precision features from pre-tokenized record tuples.

    s1_tuple   : (s1_n, s1_sq, s1_a, s1_snum, s1_c, s1_all_nums, s1_words, s1_awords)
    cand_tuple : (cn, csq, ca, csnum, cc, c_all_nums, c_words, c_awords)
    prio       : legacy argument kept for backwards compatibility (ignored)
    """
    s1_n, s1_sq, s1_a, s1_snum, s1_c, s1_nums, s1_w, s1_aw = s1_tuple
    cn, csq, ca, csnum, cc, c_nums, cw, caw = cand_tuple

    # 1. Missing Address Indicators
    both_have_addr = float(bool(s1_a and ca))
    addr_either_missing = float(not s1_a or not ca)

    # 2. Name Similarities
    jw_n = JaroWinkler.similarity(s1_n, cn)
    ts_n = rfuzz.token_sort_ratio(s1_n, cn) / 100.0
    tset_n = rfuzz.token_set_ratio(s1_n, cn) / 100.0
    ratio_n = rfuzz.ratio(s1_n, cn) / 100.0
    pr_n = rfuzz.partial_ratio(s1_n, cn) / 100.0
    exact_n = float(s1_n == cn and s1_n != '')
    name_exact_clean = float(s1_n == cn and len(s1_n) >= 5)
    exact_sq = float(s1_sq == csq and len(s1_sq) >= 4)
    jw_sq = JaroWinkler.similarity(s1_sq, csq) if (s1_sq and csq) else 0.0

    l1 = len(s1_n); l2 = len(cn)
    if l1 >= 4 and l2 >= 4:
        name_starts_with = float(s1_n.startswith(cn) or cn.startswith(s1_n))
        name_contains = float(s1_n in cn or cn in s1_n)
    else:
        name_starts_with = 0.0
        name_contains = 0.0

    len1 = len(s1_w); len2 = len(cw)
    if len1 and len2:
        set1 = set(s1_w); set2 = set(cw)
        name_jaccard = len(set1 & set2) / len(set1 | set2)
        first_tok_jw = JaroWinkler.similarity(s1_w[0], cw[0])
        last_tok_jw = JaroWinkler.similarity(s1_w[-1], cw[-1])
    else:
        name_jaccard = 0.0
        first_tok_jw = 0.0
        last_tok_jw = 0.0

    max_nl = max(l1, l2, 1)
    name_len_diff = abs(l1 - l2) / max_nl

    # 3. Address Similarities (SAFE for missing fields)
    if both_have_addr:
        jw_a = JaroWinkler.similarity(s1_a, ca)
        ts_a = rfuzz.token_sort_ratio(s1_a, ca) / 100.0
        tset_a = rfuzz.token_set_ratio(s1_a, ca) / 100.0
        exact_a = float(s1_a == ca and s1_a != '')
        if s1_aw and caw:
            aset1 = set(s1_aw); aset2 = set(caw)
            inter = len(aset1 & aset2)
            addr_jaccard = inter / len(aset1 | aset2)
            addr_containment = inter / max(min(len(aset1), len(aset2)), 1)
            addr_shared_tokens = float(inter)
            city_or_state_match = float(bool(set(s1_aw[-2:]) & set(caw[-2:])))
        else:
            addr_jaccard = 0.0
            addr_containment = 0.0
            addr_shared_tokens = 0.0
            city_or_state_match = 0.0
    else:
        # Neutral default values so missing address is NOT penalized as a conflicting address
        jw_a = 0.50
        ts_a = 0.50
        tset_a = 0.50
        exact_a = 0.0
        addr_jaccard = 0.0
        addr_containment = 0.50
        addr_shared_tokens = 0.0
        city_or_state_match = 0.50

    # 4. Street Numbers & Number Set Matching
    snum_match = float(s1_snum == csnum and s1_snum != '')
    snum_conflict = float(s1_snum != csnum and s1_snum != '' and csnum != '')

    shared_nums = len(s1_nums & c_nums) if (s1_nums and c_nums) else 0
    shared_nums_count = float(shared_nums)
    disjoint_nums = float(len(s1_nums) > 0 and len(c_nums) > 0 and shared_nums == 0)

    # 5. Cross Interactions
    cross_prod = jw_n * (jw_a if both_have_addr else 0.80)
    high_name_high_addr = float(jw_n >= 0.88 and jw_a >= 0.75 and both_have_addr)
    high_name_missing_addr = float(jw_n >= 0.88 and addr_either_missing)
    high_name_addr_conflict = float(jw_n >= 0.88 and (jw_a < 0.35 or snum_conflict) and both_have_addr)

    # 6. Country Match
    country_match = float(s1_c == cc)

    return [
        jw_n, ts_n, tset_n, ratio_n, pr_n, exact_n, name_exact_clean, exact_sq, jw_sq,
        name_starts_with, name_contains, name_jaccard, name_len_diff, first_tok_jw, last_tok_jw,
        both_have_addr, addr_either_missing, jw_a, ts_a, tset_a, exact_a, addr_jaccard,
        addr_containment, addr_shared_tokens, city_or_state_match,
        snum_match, snum_conflict, shared_nums_count, disjoint_nums,
        cross_prod, high_name_high_addr, high_name_missing_addr, high_name_addr_conflict,
        country_match
    ]

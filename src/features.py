"""
features.py
===========
High-throughput, vectorized feature engineering for Multilingual Entity Matching.

Key features:
  1. Safe missing-address handling (avoids false conflicts on NaN address)
  2. Multi-metric name similarities (Jaro-Winkler, Token Sort, Token Set, Squish)
  3. Street number matching & numerical disjointness detection
  4. Cross-attribute interactions (High Name + Missing Addr, High Name + High Addr)
  5. Country exact matching & inverted index priority scores
"""

import re
from rapidfuzz.distance import JaroWinkler
import rapidfuzz.fuzz as rfuzz

FEATURE_NAMES = [
    # 1. NAME SIMILARITIES (0-11)
    'jw_n',
    'ts_n',
    'tset_n',
    'ratio_n',
    'pr_n',
    'exact_n',
    'exact_sq',
    'jw_sq',
    'name_jaccard',
    'name_len_diff',
    'first_tok_jw',
    'last_tok_jw',

    # 2. ADDRESS SIMILARITIES & MISSING FLAGS (12-18)
    'both_have_addr',
    'addr_either_missing',
    'jw_a',
    'ts_a',
    'tset_a',
    'exact_a',
    'addr_jaccard',

    # 3. NUMBERS & STREET NUMBERS (19-22)
    'snum_match',
    'snum_conflict',
    'shared_nums_count',
    'disjoint_nums',

    # 4. CROSS INTERACTIONS (23-26)
    'cross_prod',
    'high_name_high_addr',
    'high_name_missing_addr',
    'high_name_addr_conflict',

    # 5. COUNTRY & PRIORITY (27-28)
    'country_match',
    'prio_score'
]

def compute_pair_features(s1_tuple, cand_tuple, prio: float = 0.0) -> list:
    """
    Extract 29 high-precision features from pre-tokenized record tuples.

    s1_tuple   : (s1_n, s1_sq, s1_a, s1_snum, s1_c, s1_all_nums, s1_words, s1_awords)
    cand_tuple : (cn, csq, ca, csnum, cc, c_all_nums, c_words, c_awords)
    prio       : float inverted index tier score (e.g. 100, 95, 80, ...)
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
    exact_sq = float(s1_sq == csq and len(s1_sq) >= 4)
    jw_sq = JaroWinkler.similarity(s1_sq, csq) if (s1_sq and csq) else 0.0

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

    max_nl = max(len(s1_n), len(cn), 1)
    name_len_diff = abs(len(s1_n) - len(cn)) / max_nl

    # 3. Address Similarities (SAFE for missing fields)
    if both_have_addr:
        jw_a = JaroWinkler.similarity(s1_a, ca)
        ts_a = rfuzz.token_sort_ratio(s1_a, ca) / 100.0
        tset_a = rfuzz.token_set_ratio(s1_a, ca) / 100.0
        exact_a = float(s1_a == ca and s1_a != '')
        if s1_aw and caw:
            aset1 = set(s1_aw); aset2 = set(caw)
            addr_jaccard = len(aset1 & aset2) / len(aset1 | aset2)
        else:
            addr_jaccard = 0.0
    else:
        # Neutral default values so missing address is NOT penalized as a conflicting address
        jw_a = 0.50
        ts_a = 0.50
        tset_a = 0.50
        exact_a = 0.0
        addr_jaccard = 0.0

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
    high_name_addr_conflict = float(jw_n >= 0.88 and jw_a < 0.35 and both_have_addr)

    # 6. Country Match & Priority Score
    country_match = float(s1_c == cc)

    return [
        jw_n, ts_n, tset_n, ratio_n, pr_n, exact_n, exact_sq, jw_sq, name_jaccard, name_len_diff, first_tok_jw, last_tok_jw,
        both_have_addr, addr_either_missing, jw_a, ts_a, tset_a, exact_a, addr_jaccard,
        snum_match, snum_conflict, shared_nums_count, disjoint_nums,
        cross_prod, high_name_high_addr, high_name_missing_addr, high_name_addr_conflict,
        country_match, float(prio)
    ]

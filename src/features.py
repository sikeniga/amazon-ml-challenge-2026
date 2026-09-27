"""
features.py
===========
High-throughput, vectorized feature engineering for Multilingual Entity Matching.
Implements the comprehensive feature set requested for the LightGBM Precision Strategy:
  1. Name Similarities (Exact, Normalized, RapidFuzz edit, Token Jaccard, Char n-grams)
  2. Address Similarities (Exact, JaroWinkler, Token overlap, Containment, Length diff)
  3. Structured Location (Country, City, State, PIN/ZIP, Building/Street number)
  4. Explicit Interaction Features (prod, sum, min, max)
  5. Hard Negative Indicators (High Name + Addr/PIN Conflict)
"""

import re
from rapidfuzz.distance import JaroWinkler
import rapidfuzz.fuzz as rfuzz

FEATURE_NAMES = [
    # 1. NAME SIMILARITIES (0-15)
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
    'char_ngram_jaccard',
    'name_len_diff',
    'first_tok_jw',
    'last_tok_jw',

    # 2. ADDRESS SIMILARITIES & MISSING FLAGS (16-25)
    'both_have_addr',
    'addr_either_missing',
    'jw_a',
    'ts_a',
    'tset_a',
    'exact_a',
    'addr_jaccard',
    'addr_containment',
    'addr_shared_tokens',
    'addr_len_diff',

    # 3. STRUCTURED & LOCATION ATTRIBUTES (26-35)
    'country_match',
    'city_match',
    'state_match',
    'city_or_state_match',
    'pin_match',
    'pin_conflict',
    'snum_match',
    'snum_conflict',
    'shared_nums_count',
    'disjoint_nums',

    # 4. INTERACTION FEATURES (36-42)
    'sim_interaction_prod',
    'sim_interaction_sum',
    'sim_interaction_min',
    'sim_interaction_max',
    'high_name_high_addr',
    'high_name_missing_addr',
    'high_name_addr_conflict'
]


def compute_pair_features(s1_tuple, cand_tuple, prio: float = 0.0) -> list:
    """
    Extract 43 high-precision features from pre-tokenized record tuples.

    Tuple structure:
    (name, squish, addr, street_num, country, nums_set, words, addr_words, pin, state, city, ngrams)
    """
    # Unpack pre-tokenized tuples (safe unpack supporting both 8-tuple and 12-tuple)
    if len(s1_tuple) >= 12:
        s1_n, s1_sq, s1_a, s1_snum, s1_c, s1_nums, s1_w, s1_aw, s1_pin, s1_state, s1_city, s1_ng = s1_tuple[:12]
    else:
        s1_n, s1_sq, s1_a, s1_snum, s1_c, s1_nums, s1_w, s1_aw = s1_tuple[:8]
        s1_pin, s1_state, s1_city, s1_ng = '', '', '', set()

    if len(cand_tuple) >= 12:
        cn, csq, ca, csnum, cc, c_nums, cw, caw, c_pin, c_state, c_city, c_ng = cand_tuple[:12]
    else:
        cn, csq, ca, csnum, cc, c_nums, cw, caw = cand_tuple[:8]
        c_pin, c_state, c_city, c_ng = '', '', '', set()

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

    # Character n-gram Jaccard
    if s1_ng and c_ng:
        char_ngram_jaccard = len(s1_ng & c_ng) / len(s1_ng | c_ng)
    else:
        char_ngram_jaccard = jw_sq

    max_nl = max(l1, l2, 1)
    name_len_diff = abs(l1 - l2) / max_nl

    # 3. Address Similarities
    al1 = len(s1_a); al2 = len(ca)
    addr_len_diff = abs(al1 - al2) / max(al1, al2, 1)

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
        else:
            addr_jaccard = 0.0
            addr_containment = 0.0
            addr_shared_tokens = 0.0
    else:
        jw_a = 0.50
        ts_a = 0.50
        tset_a = 0.50
        exact_a = 0.0
        addr_jaccard = 0.0
        addr_containment = 0.50
        addr_shared_tokens = 0.0

    # 4. Structured Location Attributes
    country_match = float(s1_c == cc)
    
    city_match = float(bool(s1_city and c_city and s1_city == c_city))
    state_match = float(bool(s1_state and c_state and s1_state == c_state))
    city_or_state_match = float(city_match or state_match or (bool(set(s1_aw[-2:]) & set(caw[-2:])) if (s1_aw and caw) else False))

    pin_match = float(bool(s1_pin and c_pin and s1_pin == c_pin))
    pin_conflict = float(bool(s1_pin and c_pin and s1_pin != c_pin))

    snum_match = float(bool(s1_snum and csnum and s1_snum == csnum))
    snum_conflict = float(bool(s1_snum and csnum and s1_snum != csnum))

    shared_nums = len(s1_nums & c_nums) if (s1_nums and c_nums) else 0
    shared_nums_count = float(shared_nums)
    disjoint_nums = float(len(s1_nums) > 0 and len(c_nums) > 0 and shared_nums == 0)

    # 5. Explicit Interaction Features
    eff_jw_a = jw_a if both_have_addr else 0.80
    sim_interaction_prod = jw_n * eff_jw_a
    sim_interaction_sum = (jw_n + eff_jw_a) / 2.0
    sim_interaction_min = min(jw_n, eff_jw_a)
    sim_interaction_max = max(jw_n, eff_jw_a)

    high_name_high_addr = float(jw_n >= 0.88 and jw_a >= 0.75 and both_have_addr)
    high_name_missing_addr = float(jw_n >= 0.88 and addr_either_missing)
    # Hard negative flag: identical name but conflicting location/PIN/street
    high_name_addr_conflict = float(jw_n >= 0.85 and (snum_conflict or pin_conflict or (both_have_addr and jw_a < 0.35)))

    return [
        jw_n, ts_n, tset_n, ratio_n, pr_n, exact_n, name_exact_clean, exact_sq, jw_sq,
        name_starts_with, name_contains, name_jaccard, char_ngram_jaccard, name_len_diff, first_tok_jw, last_tok_jw,
        both_have_addr, addr_either_missing, jw_a, ts_a, tset_a, exact_a, addr_jaccard,
        addr_containment, addr_shared_tokens, addr_len_diff,
        country_match, city_match, state_match, city_or_state_match, pin_match, pin_conflict,
        snum_match, snum_conflict, shared_nums_count, disjoint_nums,
        sim_interaction_prod, sim_interaction_sum, sim_interaction_min, sim_interaction_max,
        high_name_high_addr, high_name_missing_addr, high_name_addr_conflict
    ]

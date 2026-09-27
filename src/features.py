import re
import unicodedata
from rapidfuzz.distance import JaroWinkler, Levenshtein
import rapidfuzz.fuzz as rfuzz

FEATURE_NAMES = [
    # NAME FEATURES (1-14)
    'name_exact',
    'name_normalized_exact',
    'name_length_difference',
    'name_levenshtein_similarity',
    'name_jaro_winkler_similarity',
    'name_ratio',
    'name_token_sort_ratio',
    'name_token_set_ratio',
    'name_char3_similarity',
    'name_char4_similarity',
    'name_char5_similarity',
    'name_jaccard',
    'first_token_similarity',
    'last_token_similarity',

    # ADDRESS FEATURES (15-28)
    'address_exact',
    'address_length_difference',
    'address_levenshtein_similarity',
    'address_jaro_winkler_similarity',
    'address_ratio',
    'address_token_sort_ratio',
    'address_token_set_ratio',
    'address_char3_similarity',
    'address_char4_similarity',
    'address_char5_similarity',
    'address_jaccard',
    'house_number_match',
    'postal_code_match',
    'shared_number_count',

    # COUNTRY (29)
    'country_exact_match',

    # CROSS FEATURES (30-37)
    'cross_prod',
    'cross_sum',
    'cross_diff',
    'cross_min',
    'cross_max',
    'high_name_high_address',
    'high_name_missing_address',
    'high_name_address_conflict',

    # BLOCKING PRIO (38)
    'prio_score'
]

def clean_name(s: str) -> str:
    if not isinstance(s, str) or not s: return ''
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8').lower()
    for sep in [' d/b/a ', ' dba ', ' t/a ', ' ta ', ' trading as ']:
        if sep in s: s = s.split(sep)[-1]; break
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\b(inc|corp|corporation|incorporated|llc|pllc|ltd|limited|co|company|pvt|private|llp|pc|sarl|sas|sasu|sa|eurl|snc|sci|gie)\b', ' ', s)
    return ' '.join(s.split())

def squish(s: str) -> str:
    s = re.sub(r'\b(com|org|net|in|fr|io|co|biz|info)\b', '', s)
    return s.replace(' ', '')

def clean_addr(s: str) -> str:
    if not isinstance(s, str) or not s: return ''
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8').lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\b(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|way|suite|ste|apt|floor|fl)\b', ' ', s)
    return ' '.join(s.split())

def get_street_num(addr: str) -> str:
    if not isinstance(addr, str) or not addr: return ''
    nums = re.findall(r'\b\d+\b', addr)
    return nums[0] if nums else ''

def get_postal_code(addr: str) -> str:
    if not isinstance(addr, str) or not addr: return ''
    postals = re.findall(r'\b\d{5}(?:-\d{4})?\b|\b[a-z]\d[a-z]\s?\d[a-z]\d\b', addr.lower())
    return postals[0].replace(' ', '') if postals else ''

def get_street_prefix(addr: str) -> str:
    if not isinstance(addr, str) or not addr: return ''
    ca = clean_name(addr)
    words = [w for w in ca.split() if not w.isdigit() and len(w) >= 3]
    return words[0][:3] if words else ''

def _char_ngrams(s: str, n: int) -> set:
    if len(s) < n: return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b: return 1.0
    if not set_a or not set_b: return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def extract_pair_features(
    s1_raw_name: str, s1_raw_addr: str, s1_ctry: str,
    c_raw_name: str, c_raw_addr: str, c_ctry: str,
    prio_score: float = 0.0
) -> list:
    s1_n = clean_name(s1_raw_name)
    cn = clean_name(c_raw_name)
    s1_a = clean_addr(s1_raw_addr)
    ca = clean_addr(c_raw_addr)

    s1_c = str(s1_ctry).upper().strip()
    cc = str(c_ctry).upper().strip()

    s1_w = s1_n.split()
    cw = cn.split()
    s1_aw = s1_a.split()
    caw = ca.split()

    # 1. NAME FEATURES
    name_exact = float(s1_raw_name == c_raw_name and s1_raw_name != '')
    name_normalized_exact = float(s1_n == cn and s1_n != '')
    name_length_diff = float(abs(len(s1_n) - len(cn)))
    
    max_nl = max(len(s1_n), len(cn), 1)
    name_lev = 1.0 - (Levenshtein.distance(s1_n, cn) / max_nl)
    name_jw = JaroWinkler.similarity(s1_n, cn)
    name_ratio = rfuzz.ratio(s1_n, cn) / 100.0
    name_ts = rfuzz.token_sort_ratio(s1_n, cn) / 100.0
    name_tset = rfuzz.token_set_ratio(s1_n, cn) / 100.0

    name_char3 = _jaccard(_char_ngrams(s1_n, 3), _char_ngrams(cn, 3))
    name_char4 = _jaccard(_char_ngrams(s1_n, 4), _char_ngrams(cn, 4))
    name_char5 = _jaccard(_char_ngrams(s1_n, 5), _char_ngrams(cn, 5))
    name_jaccard = _jaccard(set(s1_w), set(cw))

    first_tok_sim = JaroWinkler.similarity(s1_w[0], cw[0]) if (s1_w and cw) else 0.0
    last_tok_sim = JaroWinkler.similarity(s1_w[-1], cw[-1]) if (s1_w and cw) else 0.0

    # 2. ADDRESS FEATURES
    addr_exact = float(s1_raw_addr == c_raw_addr and s1_raw_addr != '')
    addr_length_diff = float(abs(len(s1_a) - len(ca)))
    
    max_al = max(len(s1_a), len(ca), 1)
    addr_lev = 1.0 - (Levenshtein.distance(s1_a, ca) / max_al) if (s1_a and ca) else 0.0
    addr_jw = JaroWinkler.similarity(s1_a, ca) if (s1_a and ca) else 0.0
    addr_ratio = (rfuzz.ratio(s1_a, ca) / 100.0) if (s1_a and ca) else 0.0
    addr_ts = (rfuzz.token_sort_ratio(s1_a, ca) / 100.0) if (s1_a and ca) else 0.0
    addr_tset = (rfuzz.token_set_ratio(s1_a, ca) / 100.0) if (s1_a and ca) else 0.0

    addr_char3 = _jaccard(_char_ngrams(s1_a, 3), _char_ngrams(ca, 3)) if (s1_a and ca) else 0.0
    addr_char4 = _jaccard(_char_ngrams(s1_a, 4), _char_ngrams(ca, 4)) if (s1_a and ca) else 0.0
    addr_char5 = _jaccard(_char_ngrams(s1_a, 5), _char_ngrams(ca, 5)) if (s1_a and ca) else 0.0
    addr_jaccard = _jaccard(set(s1_aw), set(caw)) if (s1_aw and caw) else 0.0

    s1_snum = get_street_num(s1_raw_addr)
    csnum = get_street_num(c_raw_addr)
    house_match = float(s1_snum == csnum and s1_snum != '')

    s1_post = get_postal_code(s1_raw_addr)
    cpost = get_postal_code(c_raw_addr)
    postal_match = float(s1_post == cpost and s1_post != '')

    s1_nums = set(re.findall(r'\b\d+\b', s1_raw_addr))
    c_nums = set(re.findall(r'\b\d+\b', c_raw_addr))
    shared_num_count = float(len(s1_nums & c_nums))

    # 3. COUNTRY FEATURE
    country_exact = float(s1_c == cc)

    # 4. CROSS FEATURES
    cross_prod = name_jw * addr_jw
    cross_sum = name_jw + addr_jw
    cross_diff = name_jw - addr_jw
    cross_min = min(name_jw, addr_jw)
    cross_max = max(name_jw, addr_jw)

    high_name_high_addr = float(name_jw >= 0.85 and addr_jw >= 0.80)
    high_name_missing_addr = float(name_jw >= 0.85 and (s1_a == '' or ca == ''))
    high_name_addr_conflict = float(name_jw >= 0.85 and addr_jw < 0.40 and s1_a != '' and ca != '')

    return [
        name_exact,
        name_normalized_exact,
        name_length_diff,
        name_lev,
        name_jw,
        name_ratio,
        name_ts,
        name_tset,
        name_char3,
        name_char4,
        name_char5,
        name_jaccard,
        first_tok_sim,
        last_tok_sim,
        addr_exact,
        addr_length_diff,
        addr_lev,
        addr_jw,
        addr_ratio,
        addr_ts,
        addr_tset,
        addr_char3,
        addr_char4,
        addr_char5,
        addr_jaccard,
        house_match,
        postal_match,
        shared_num_count,
        country_exact,
        cross_prod,
        cross_sum,
        cross_diff,
        cross_min,
        cross_max,
        high_name_high_addr,
        high_name_missing_addr,
        high_name_addr_conflict,
        float(prio_score)
    ]

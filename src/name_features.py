"""
name_features.py
================
String similarity features for business name pairs.

Features computed per pair:
  name_exact            : 1 if normalized names are identical
  name_levenshtein      : 1 - edit_distance / max_len
  name_jaccard_token    : Jaccard on token sets (stopwords removed)
  name_jaccard_char2    : Jaccard on character 2-gram sets
  name_jaccard_char3    : Jaccard on character 3-gram sets
  name_token_overlap    : |intersection| / max(|set_a|, |set_b|)
  name_tfidf_cosine     : precomputed TF-IDF cosine (from features.py)
  name_len_diff         : |len1-len2| / max_len  (0=same length)
  name_first_token_match: 1 if first tokens match
"""


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = curr
    return prev[-1]


def levenshtein_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein_distance(a, b) / max_len


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def char_ngrams(text: str, n: int) -> set:
    return {text[i: i + n] for i in range(len(text) - n + 1)}


def token_overlap(tokens_a: list, tokens_b: list) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    s_a, s_b = set(tokens_a), set(tokens_b)
    max_len = max(len(s_a), len(s_b))
    if max_len == 0:
        return 0.0
    return len(s_a & s_b) / max_len


def name_features(
    norm_name1: str,
    norm_name2: str,
    tokens1: list,
    tokens2: list,
    tfidf_cosine: float = 0.0,
) -> dict:
    """
    Compute all name similarity features for a single candidate pair.

    Parameters
    ----------
    norm_name1, norm_name2 : normalized (lowercased, abbrev-expanded) name strings
    tokens1, tokens2       : tokenized name lists (stopwords removed)
    tfidf_cosine           : precomputed TF-IDF cosine similarity
    """
    exact = int(norm_name1 == norm_name2)
    lev = levenshtein_sim(norm_name1, norm_name2)
    jac_tok = jaccard(set(tokens1), set(tokens2))
    jac_c2 = jaccard(char_ngrams(norm_name1, 2), char_ngrams(norm_name2, 2))
    jac_c3 = jaccard(char_ngrams(norm_name1, 3), char_ngrams(norm_name2, 3))
    tok_ov = token_overlap(tokens1, tokens2)

    max_len = max(len(norm_name1), len(norm_name2)) or 1
    len_diff = abs(len(norm_name1) - len(norm_name2)) / max_len

    ft1 = tokens1[0] if tokens1 else ""
    ft2 = tokens2[0] if tokens2 else ""
    first_tok_match = int(ft1 == ft2 and ft1 != "")

    return {
        "name_exact": exact,
        "name_levenshtein": lev,
        "name_jaccard_token": jac_tok,
        "name_jaccard_char2": jac_c2,
        "name_jaccard_char3": jac_c3,
        "name_token_overlap": tok_ov,
        "name_tfidf_cosine": tfidf_cosine,
        "name_len_diff": len_diff,
        "name_first_token_match": first_tok_match,
    }

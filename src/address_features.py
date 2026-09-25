"""
address_features.py
===================
String similarity features for business address pairs.

Features per candidate pair:
  address_exact         : 1 if normalized addresses are identical
  address_levenshtein   : char edit similarity (capped at 100 chars for speed)
  address_jaccard_token : Jaccard on address token sets
  address_jaccard_char2 : Jaccard on 2-gram sets
  address_jaccard_char3 : Jaccard on 3-gram sets
  address_token_overlap : shared token count / max token set size
  address_tfidf_cosine  : precomputed TF-IDF cosine (from features.py)
  address_len_diff      : normalized length difference
"""

from name_features import jaccard, char_ngrams, token_overlap, levenshtein_sim


def address_features(
    norm_addr1: str,
    norm_addr2: str,
    tokens1: list,
    tokens2: list,
    tfidf_cosine: float = 0.0,
) -> dict:
    """
    Compute all address similarity features for a single candidate pair.

    Parameters
    ----------
    norm_addr1, norm_addr2 : normalized address strings
    tokens1, tokens2       : tokenized address lists
    tfidf_cosine           : precomputed TF-IDF cosine similarity
    """
    exact = int(norm_addr1 == norm_addr2)
    # Cap address length for Levenshtein to avoid O(n^2) slowdown on long addresses
    lev = levenshtein_sim(norm_addr1[:100], norm_addr2[:100])
    jac_tok = jaccard(set(tokens1), set(tokens2))
    jac_c2 = jaccard(char_ngrams(norm_addr1, 2), char_ngrams(norm_addr2, 2))
    jac_c3 = jaccard(char_ngrams(norm_addr1, 3), char_ngrams(norm_addr2, 3))
    tok_ov = token_overlap(tokens1, tokens2)

    max_len = max(len(norm_addr1), len(norm_addr2)) or 1
    len_diff = abs(len(norm_addr1) - len(norm_addr2)) / max_len

    return {
        "address_exact": exact,
        "address_levenshtein": lev,
        "address_jaccard_token": jac_tok,
        "address_jaccard_char2": jac_c2,
        "address_jaccard_char3": jac_c3,
        "address_token_overlap": tok_ov,
        "address_tfidf_cosine": tfidf_cosine,
        "address_len_diff": len_diff,
    }

"""
preprocessing.py
================
Business name and address normalization utilities.

Key responsibilities:
  - Clean and normalize business names (abbreviation expansion, punctuation, lower-case)
  - Clean and normalize addresses (abbreviation expansion, token sorting)
  - Provide tokenized versions for downstream similarity features

Performance notes (fixed):
  - _apply_abbrev used to run one re.sub() PER abbreviation (30 for names, 24 for
    addresses), each scanning the full string. That is 30/24 full passes per row,
    which does not scale to millions of rows. It is now a single compiled
    alternation regex -> one pass per string.
  - normalize_name/normalize_address used to rely on functools.lru_cache(32768).
    With 2M+ unique business names, a cache that small gets evicted constantly
    and provides little benefit. preprocess_dataframe now normalizes only the
    *unique* values in each column and maps the results back, which is a much
    bigger win than lru_cache at this cardinality (repeats are common, but the
    unique count still exceeds a small LRU cache).
"""

import re
import unicodedata
from functools import lru_cache

# ---------------------------------------------------------------------------
# Abbreviation maps
# ---------------------------------------------------------------------------

NAME_ABBREV = {
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\binc\b": "incorporated",
    r"\bltd\b": "limited",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\bpvt\b": "private",
    r"\bpte\b": "private",
    r"\bsdn\b": "sendirian",
    r"\bbhd\b": "berhad",
    r"\bplc\b": "public limited company",
    r"\bgmbh\b": "gesellschaft mit beschrankter haftung",
    r"\bsa\b": "societe anonyme",
    r"\bsas\b": "societe par actions simplifiee",
    r"\bsarl\b": "societe a responsabilite limitee",
    r"\bintl\b": "international",
    r"\bint l\b": "international",
    r"\bmfg\b": "manufacturing",
    r"\bsvcs\b": "services",
    r"\bsvc\b": "service",
    r"\bmgmt\b": "management",
    r"\bassoc\b": "associates",
    r"\bassn\b": "association",
    r"\bdept\b": "department",
    r"\bgrp\b": "group",
    r"\btech\b": "technology",
    r"\bsoln\b": "solution",
    r"\bsolns\b": "solutions",
    r"\benterprises\b": "enterprise",
    r"\bnatl\b": "national",
}

ADDRESS_ABBREV = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bct\b": "court",
    r"\bln\b": "lane",
    r"\bpl\b": "place",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bfwy\b": "freeway",
    r"\bsq\b": "square",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bflr\b": "floor",
    r"\bbldg\b": "building",
    r"\bctr\b": "center",
    r"\bjn\b": "junction",
    r"\bnear\b": "near",
    r"\bopp\b": "opposite",
    r"\bnh\b": "national highway",
    r"\bsh\b": "state highway",
    r"\bno\b": "number",
}

STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "in", "at", "by", "to", "is",
    "on", "with", "de", "la", "le", "les", "du", "des", "van", "der",
}

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _unicode_normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.encode("ascii", "ignore").decode("ascii")


def _compile_abbrev(abbrev_map: dict):
    """
    Combine many `\\bword\\b -> replacement` rules into ONE compiled
    alternation regex, so each string is scanned once instead of once
    per abbreviation.

    Multi-word patterns like `\\bint l\\b` are supported: the literal
    inner text (with internal whitespace normalized to a single space)
    is used as the alternative and matched as-is.
    """
    lookup = {}
    parts = []
    for pattern, replacement in abbrev_map.items():
        # Strip a leading/trailing \b and normalize internal whitespace
        # (handles both "\bword\b" and "\bmulti word\b" cases).
        word = pattern
        if word.startswith(r"\b"):
            word = word[2:]
        if word.endswith(r"\b"):
            word = word[:-2]
        word = re.sub(r"\s+", " ", word).strip()
        lookup[word] = replacement
        parts.append(re.escape(word).replace(r"\ ", r"\s+"))

    # Longer alternatives first so multi-word patterns aren't shadowed
    # by a shorter single-word prefix.
    parts.sort(key=len, reverse=True)
    combined = re.compile(r"\b(" + "|".join(parts) + r")\b")

    def _sub(text: str) -> str:
        def _repl(m: "re.Match") -> str:
            key = re.sub(r"\s+", " ", m.group(0)).strip()
            return lookup.get(key, m.group(0))
        return combined.sub(_repl, text)

    return _sub


_apply_name_abbrev = _compile_abbrev(NAME_ABBREV)
_apply_address_abbrev = _compile_abbrev(ADDRESS_ABBREV)


def _apply_abbrev(text: str, abbrev_map: dict) -> str:
    """Kept for backwards compatibility / debugging (slow path, not used
    internally anymore)."""
    for pattern, replacement in abbrev_map.items():
        text = re.sub(pattern, replacement, text)
    return text


# ---------------------------------------------------------------------------
# Business name normalization
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32768)
def normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    text = _unicode_normalize(name)
    text = text.lower()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"[-']", " ", text)
    text = _apply_name_abbrev(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_name(name: str, remove_stopwords: bool = True):
    tokens = normalize_name(name).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def name_first_token(name: str) -> str:
    tokens = tokenize_name(name, remove_stopwords=False)
    return tokens[0] if tokens else ""


def name_first_char(name: str) -> str:
    n = normalize_name(name)
    return n[0] if n else ""


def name_prefix(name: str, length: int = 3) -> str:
    n = normalize_name(name)
    return n[:length]


# ---------------------------------------------------------------------------
# Address normalization
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32768)
def normalize_address(address: str) -> str:
    if not isinstance(address, str) or not address.strip():
        return ""
    text = _unicode_normalize(address)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = _apply_address_abbrev(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_address(address: str):
    return normalize_address(address).split()


def address_first_token(address: str) -> str:
    tokens = tokenize_address(address)
    return tokens[0] if tokens else ""


# ---------------------------------------------------------------------------
# Country normalization
# ---------------------------------------------------------------------------

COUNTRY_MAP = {
    "usa": "us",
    "united states": "us",
    "united states of america": "us",
    "america": "us",
    "india": "in",
    "ind": "in",
    "bharat": "in",
    "france": "fr",
    "french republic": "fr",
    "republique francaise": "fr",
}


def normalize_country(country: str) -> str:
    if not isinstance(country, str):
        return ""
    c = country.lower().strip()
    return COUNTRY_MAP.get(c, c)


# ---------------------------------------------------------------------------
# DataFrame-level application
# ---------------------------------------------------------------------------

def _map_via_unique(series, func, label=""):
    """
    Apply `func` to only the unique values of `series`, then map results
    back. Much faster than series.map(func) directly when there are many
    repeated values and/or the value cardinality exceeds a small
    lru_cache, since each distinct value is computed exactly once
    regardless of cache size or eviction.
    """
    uniques = series.unique()
    if label:
        print(f"    {label}: {len(series):,} rows -> {len(uniques):,} unique values")
    mapping = {v: func(v) for v in uniques}
    return series.map(mapping)


def preprocess_dataframe(df, verbose: bool = True):
    df = df.copy()
    n = len(df)
    if verbose:
        print(f"  preprocessing {n:,} rows...")

    business_name = df["business_name"].fillna("")
    business_address = df["business_address"].fillna("")
    country = df["country"].fillna("")

    df["norm_name"] = _map_via_unique(
        business_name, normalize_name, "norm_name" if verbose else ""
    )
    df["norm_address"] = _map_via_unique(
        business_address, normalize_address, "norm_address" if verbose else ""
    )
    df["norm_country"] = _map_via_unique(
        country, normalize_country, "norm_country" if verbose else ""
    )

    df["name_tokens"] = _map_via_unique(
        df["norm_name"],
        lambda x: tokenize_name(x, remove_stopwords=True),
        "name_tokens" if verbose else "",
    )
    df["address_tokens"] = _map_via_unique(
        df["norm_address"], tokenize_address, "address_tokens" if verbose else ""
    )
    df["name_first_token"] = _map_via_unique(
        df["norm_name"], name_first_token, "name_first_token" if verbose else ""
    )
    df["name_prefix3"] = _map_via_unique(
        df["norm_name"], lambda x: name_prefix(x, 3), "name_prefix3" if verbose else ""
    )
    df["name_first_char"] = _map_via_unique(
        df["norm_name"], name_first_char, "name_first_char" if verbose else ""
    )

    if verbose:
        print(f"  done preprocessing {n:,} rows.")
    return df
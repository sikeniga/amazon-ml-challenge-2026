"""
preprocessing.py
================
Business name and address normalization utilities.

Key responsibilities:
  - Clean and normalize business names (abbreviation expansion, punctuation, lower-case)
  - Clean and normalize addresses (abbreviation expansion, token sorting)
  - Provide tokenized versions for downstream similarity features
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
    r"\b&\b": "and",
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


def _apply_abbrev(text: str, abbrev_map: dict) -> str:
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
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"[-']", " ", text)
    text = _apply_abbrev(text, NAME_ABBREV)
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
    text = _apply_abbrev(text, ADDRESS_ABBREV)
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

def preprocess_dataframe(df):
    df = df.copy()
    df["norm_name"] = df["business_name"].fillna("").map(normalize_name)
    df["norm_address"] = df["business_address"].fillna("").map(normalize_address)
    df["norm_country"] = df["country"].fillna("").map(normalize_country)
    df["name_tokens"] = df["norm_name"].map(lambda x: tokenize_name(x, remove_stopwords=True))
    df["address_tokens"] = df["norm_address"].map(tokenize_address)
    df["name_first_token"] = df["norm_name"].map(name_first_token)
    df["name_prefix3"] = df["norm_name"].map(lambda x: name_prefix(x, 3))
    df["name_first_char"] = df["norm_name"].map(name_first_char)
    return df

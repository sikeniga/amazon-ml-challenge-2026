"""
preprocessing.py
================
Enterprise-grade normalization for Multilingual Business Entity Resolution
(Amazon ML Challenge 2026).

Handles:
  1. AnyAscii transliteration (Indic scripts, European diacritics, Cyrillic, etc.)
  2. Web prefixes & domain suffixes (www., http://, .com, .in, .org, .fr, etc.)
  3. DBA / Trade name variations (d/b/a, dba, t/a, ta, trading as)
  4. Honorific / Salutation prefixes (Dr, Sri, Shree, Shri, M/s, Mr, Mrs, Prof, Er, CA)
  5. Legal entity designations (English, French, Transliterated Indic: LLP, Pvt Ltd, SARL, SAS, elelpi, praivet, etc.)
  6. Country-specific address normalization (Indian states/cities, French street types)
"""

import re
import anyascii

# ---------------------------------------------------------------------------
# Country-Specific Synonym Dictionaries
# ---------------------------------------------------------------------------

INDIAN_STATE_SYNONYMS = {
    r'\bup\b': 'uttar pradesh', r'\bmh\b': 'maharashtra', r'\btn\b': 'tamil nadu',
    r'\bka\b': 'karnataka', r'\bdl\b': 'delhi', r'\bwb\b': 'west bengal',
    r'\bgj\b': 'gujarat', r'\brj\b': 'rajasthan', r'\bts\b': 'telangana',
    r'\bap\b': 'andhra pradesh', r'\bkl\b': 'kerala', r'\bhr\b': 'haryana',
    r'\bmp\b': 'madhya pradesh', r'\bpb\b': 'punjab', r'\bod\b': 'odisha',
    r'\bor\b': 'odisha', r'\bjh\b': 'jharkhand', r'\bbr\b': 'bihar',
    r'\bcg\b': 'chhattisgarh', r'\bct\b': 'chhattisgarh', r'\bga\b': 'goa',
    r'\bas\b': 'assam', r'\buk\b': 'uttarakhand', r'\bua\b': 'uttarakhand',
    r'\bjk\b': 'jammu kashmir', r'\bch\b': 'chandigarh', r'\btr\b': 'tripura',
    r'\bbengaluru\b': 'bangalore', r'\bmumbai\b': 'bombay',
    r'\bchennai\b': 'madras', r'\bkolkata\b': 'calcutta',
    r'\bvadodara\b': 'baroda', r'\bgurugram\b': 'gurgaon',
    r'\bpune\b': 'poona', r'\bkochi\b': 'cochin',
    r'\bprayagraj\b': 'allahabad', r'\bmysuru\b': 'mysore',
    r'\bvisakhapatnam\b': 'vizag', r'\bsecunderabad\b': 'hyderabad',
    r'\bpuducherry\b': 'pondicherry', r'\bthiruvananthapuram\b': 'trivandrum',
}

FRENCH_STREET_SYNONYMS = {
    r'\br\b': 'rue', r'\brue\b': 'rue',
    r'\bav\b': 'avenue', r'\bave\b': 'avenue',
    r'\bbd\b': 'boulevard', r'\bblvd\b': 'boulevard',
    r'\ball\b': 'allee', r'\ballee\b': 'allee',
    r'\bch\b': 'chemin', r'\bchemin\b': 'chemin',
    r'\bimp\b': 'impasse', r'\bimpasse\b': 'impasse',
    r'\bpl\b': 'place', r'\bplace\b': 'place',
    r'\bpass\b': 'passage', r'\bpassage\b': 'passage',
    r'\brt\b': 'route', r'\broute\b': 'route',
    r'\bquai\b': 'quai',
}

# Compiled regexes for fast execution
RE_WEB = re.compile(r'\b(https?://|www\.)', re.IGNORECASE)
RE_DOMAINS = re.compile(r'\.(com|in|org|net|co|io|fr|biz|info|gov|edu)\b', re.IGNORECASE)
RE_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')
RE_HONORIFICS = re.compile(r'^(dr|mr|mrs|ms|prof|sri|shree|shri|m/s|m\s+s|er|ca)\b\s*', re.IGNORECASE)
RE_LEGAL_SUFFIXES = re.compile(
    r'\b(inc|corp|corporation|incorporated|llc|pllc|ltd|limited|co|company|'
    r'pvt|private|llp|pc|sarl|sas|sasu|sa|eurl|snc|sci|gie|praivet|limitid|'
    r'limiteed|elelpi|pvtltd)\b',
    re.IGNORECASE
)
RE_ADDR_NOISE = re.compile(
    r'\b(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|way|'
    r'suite|ste|apt|floor|fl|near|opp|behind|beside|flat|plot|no|bldg|building|chambers|tower|'
    r'complex|nagar|colony|extn|extension|sector|sec|phase|block|blk|dist|district|mandal|po|'
    r'bazar|bazaar|marg|chowk|area|estate|industrial|indl|village|vill|city)\b',
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Normalization Functions
# ---------------------------------------------------------------------------

def clean_name(s: str) -> str:
    """
    Standardize business names:
    - Transliterate Unicode/Indic scripts to Latin ASCII
    - Strip URLs, domains, and web prefixes
    - Separate DBAs
    - Strip honorifics/salutations
    - Strip legal entity suffixes
    """
    if not isinstance(s, str) or not s:
        return ''
    
    # 1. Transliterate (e.g. Hindi, Kannada, Telugu, Tamil, French accents)
    s = anyascii.anyascii(s).lower()
    
    # 2. Strip web artifacts
    s = RE_WEB.sub('', s)
    s = RE_DOMAINS.sub('', s)
    
    # 3. Handle DBA / Trade name separators
    for sep in [' d/b/a ', ' dba ', ' t/a ', ' ta ', ' trading as ']:
        if sep in s:
            s = s.split(sep)[-1]
            break
            
    # 4. Remove punctuation
    s = RE_NON_ALPHANUM.sub(' ', s)
    
    # 5. Remove honorifics (e.g. Dr, Sri, Shree, M/s)
    s = RE_HONORIFICS.sub('', s)
    
    # 6. Remove legal suffixes (English, French, Indic transliterations)
    s = RE_LEGAL_SUFFIXES.sub(' ', s)
    
    return ' '.join(s.split())


def squish(s: str) -> str:
    """Compact slug without whitespace for robust exact/phonetic matching."""
    s = re.sub(r'\b(com|org|net|in|fr|io|co|biz|info)\b', '', s)
    return s.replace(' ', '')


def clean_addr(s: str, country: str = '') -> str:
    """
    Standardize addresses:
    - Transliterate Indic and accented characters
    - Remove common street noise words
    - Expand country-specific abbreviations
    """
    if not isinstance(s, str) or not s:
        return ''
    
    s = anyascii.anyascii(s).lower()
    s = RE_NON_ALPHANUM.sub(' ', s)
    s = RE_ADDR_NOISE.sub(' ', s)
    
    country_upper = str(country).upper().strip()
    if country_upper == 'FRANCE':
        for pat, repl in FRENCH_STREET_SYNONYMS.items():
            s = re.sub(pat, repl, s)
    elif country_upper == 'INDIA':
        for pat, repl in INDIAN_STATE_SYNONYMS.items():
            s = re.sub(pat, repl, s)
            
    return ' '.join(s.split())


def get_street_num(addr: str) -> str:
    """Extract first street number."""
    if not isinstance(addr, str) or not addr:
        return ''
    nums = re.findall(r'\b\d+\b', addr)
    return nums[0] if nums else ''


def get_all_nums(addr: str) -> set:
    """Extract all numeric tokens from address."""
    if not isinstance(addr, str) or not addr:
        return set()
    return set(re.findall(r'\b\d+\b', addr))


def get_street_prefix(addr: str) -> str:
    """Extract 3-letter prefix of first non-numeric word in address."""
    if not isinstance(addr, str) or not addr:
        return ''
    ca = clean_name(addr)
    words = [w for w in ca.split() if not w.isdigit() and len(w) >= 3 and w != 'rue']
    return words[0][:3] if words else ''


def get_distinctive_address_tokens(addr: str, country: str = '') -> list:
    """Extract long locality/city/state tokens from address."""
    if not isinstance(addr, str) or not addr:
        return []
    ca = clean_addr(addr, country)
    words = [w for w in ca.split() if not w.isdigit() and len(w) >= 4]
    return words[-4:] if len(words) >= 4 else words
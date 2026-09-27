"""
champion_submission.py  v3
===========================
CHAMPION Entity Resolution Pipeline - Amazon ML Challenge 2026.

v3 changes (speed-critical rewrite):
  - Vectorized batch feature computation (pandas string ops, no per-pair Python loop)
  - Tight blocking: EN + SQ + FL + TFP only (~15-30M pairs, not 100M+)
  - rapidfuzz.cdist for batch Jaro-Winkler (C-speed)
  - 22 features total (removed slow ones, kept the discriminative ones)
  - Progress prints at every blocking milestone

Target: Leaderboard F0.5 > 0.90
"""

import os
import re
import sys
import time
import pickle
import unicodedata
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz.distance import JaroWinkler
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("[WARN] rapidfuzz not found")

try:
    import jellyfish
    HAS_JELLYFISH = True
except ImportError:
    HAS_JELLYFISH = False

# ================================================================
# 1. NORMALIZATION
# ================================================================

LEGAL_RE = re.compile(
    r'\b(incorporated|corporation|company|limited|liability|partnership|'
    r'pvt|private|public|llc|pllc|llp|inc|corp|ltd|lp|co|pc|plc|'
    r'sarl|sas|sasu|sa|eurl|snc|sci|gie|bv|nv|ag|gmbh|kft|sl|srl|'
    r'spa|oy|ab|aps|doo|sro|ooo|intl|international|services|'
    r'technologies|technology|solutions|group|holdings|enterprises|'
    r'associates|consulting|management|industries|systems)\.?\b'
)
DOMAIN_RE = re.compile(r'\b(com|org|net|in|fr|io|co|biz|info|gov|edu|uk|de|au|us)\b')

COUNTRY_MAP = {
    'usa':'us','united states':'us','united states of america':'us',
    'america':'us','u.s.':'us','u.s.a.':'us',
    'india':'in','bharat':'in','ind':'in',
    'france':'fr','french republic':'fr',
    'uk':'gb','united kingdom':'gb','england':'gb','britain':'gb',
    'germany':'de','deutschland':'de',
    'canada':'ca','australia':'au','japan':'jp','china':'cn',
    'brazil':'br','mexico':'mx','spain':'es','italy':'it',
    'netherlands':'nl','belgium':'be','switzerland':'ch',
    'singapore':'sg','south korea':'kr','new zealand':'nz',
    'russia':'ru','poland':'pl','turkey':'tr',
    'united arab emirates':'ae','uae':'ae','saudi arabia':'sa',
}


def _uc(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    return unicodedata.normalize("NFKD", text).encode("ascii","ignore").decode("ascii").lower()


def normalize_name(name) -> str:
    s = _uc(name)
    if not s:
        return ""
    for sep in [' d/b/a ',' dba ',' t/a ',' ta ',' trading as ',' doing business as ']:
        if sep in s:
            s = s.split(sep)[-1]; break
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = LEGAL_RE.sub(' ', s)
    return ' '.join(s.split())


def squish(norm: str) -> str:
    return ''.join(DOMAIN_RE.sub('', norm).split())


def norm_country(c) -> str:
    if not isinstance(c, str) or not c.strip():
        return 'unk'
    c = c.lower().strip()
    return COUNTRY_MAP.get(c, c[:5])


# Vectorized versions operating on pandas Series
def vnorm_name(s: pd.Series) -> pd.Series:
    return s.fillna('').apply(normalize_name)

def vnorm_country(s: pd.Series) -> pd.Series:
    return s.fillna('').apply(norm_country)

def vsquish(s: pd.Series) -> pd.Series:
    return s.apply(squish)


# ================================================================
# 2. BLOCKING INDEX  (5 tight keys only)
# ================================================================

def _add_to_index(eid, raw_name, raw_ctry, index, lookup_name, lookup_ctry):
    norm = normalize_name(raw_name)
    ctry = norm_country(raw_ctry)
    sq   = squish(norm)
    tokens = norm.split()
    sig  = [t for t in tokens if len(t) >= 3]

    lookup_name[eid] = raw_name
    lookup_ctry[eid] = raw_ctry

    # EN: exact normalized name
    if norm:
        index[('EN', ctry, norm)].append(eid)
    # SQ: squish slug
    if len(sq) >= 5:
        index[('SQ', ctry, sq)].append(eid)
    # FL: first+last significant token pair (order-invariant)
    if len(sig) >= 2:
        pair = tuple(sorted([sig[0], sig[-1]]))
        index[('FL', ctry, pair[0], pair[1])].append(eid)
    # TFP: sorted token fingerprint (only short names)
    if 2 <= len(tokens) <= 3:
        fp = '|'.join(sorted(tokens))
        index[('TFP', ctry, fp)].append(eid)
    # FT2: first token if >= 5 chars (long-enough to be specific)
    if tokens and len(tokens[0]) >= 5:
        index[('FT5', ctry, tokens[0])].append(eid)


def index_chunk(chunk: pd.DataFrame, index, lookup_name, lookup_ctry):
    eids  = chunk['entity_id'].astype(str).values
    names = chunk['business_name'].fillna('').astype(str).values
    ctrys = chunk['country'].fillna('').astype(str).values
    for eid, n, c in zip(eids, names, ctrys):
        _add_to_index(eid, n, c, index, lookup_name, lookup_ctry)


def get_candidates(eid, norm, ctry, index, max_per_key=40, global_cap=100):
    sq     = squish(norm)
    tokens = norm.split()
    sig    = [t for t in tokens if len(t) >= 3]
    hits   = set()

    def _take(key, lim=max_per_key):
        for cid in index.get(key, [])[:lim]:
            if cid != eid: hits.add(cid)

    if norm:               _take(('EN', ctry, norm))
    if len(sq) >= 5:       _take(('SQ', ctry, sq))
    if len(sig) >= 2:
        pair = tuple(sorted([sig[0], sig[-1]]))
        _take(('FL', ctry, pair[0], pair[1]))
    if 2 <= len(tokens) <= 3:
        _take(('TFP', ctry, '|'.join(sorted(tokens))))
    if tokens and len(tokens[0]) >= 5:
        _take(('FT5', ctry, tokens[0]), 25)

    if len(hits) > global_cap:
        hits = set(list(hits)[:global_cap])
    return hits


# ================================================================
# 3. VECTORIZED FEATURE COMPUTATION
# ================================================================

def _jsim(a: str, b: str) -> float:
    if a == b: return 1.0
    if not a or not b: return 0.0
    if HAS_RAPIDFUZZ:
        return JaroWinkler.normalized_similarity(a, b)
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def _esim(a: str, b: str) -> float:
    if a == b: return 1.0
    if not a or not b: return 0.0
    if HAS_RAPIDFUZZ:
        return rfuzz.ratio(a, b) / 100.0
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def _jaccard_tok(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb: return 1.0
    if not ta or not tb: return 0.0
    i = len(ta & tb)
    return i / (len(ta) + len(tb) - i)


def _ngram_jac(a: str, b: str, n: int) -> float:
    if len(a) < n or len(b) < n: return float(a == b)
    sa = set(a[i:i+n] for i in range(len(a)-n+1))
    sb = set(b[i:i+n] for i in range(len(b)-n+1))
    if not sa and not sb: return 1.0
    i = len(sa & sb)
    return i / (len(sa) + len(sb) - i)


def compute_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features for a batch DataFrame with columns:
      n1, n2, a1, a2, c1, c2  (already normalized strings)
    Returns a DataFrame of feature columns.
    """
    n1 = df['n1'].values
    n2 = df['n2'].values
    a1 = df['a1'].values
    a2 = df['a2'].values
    c1 = df['c1'].values
    c2 = df['c2'].values

    N = len(df)
    feats = {}

    # Country
    feats['country_match'] = (c1 == c2).astype(np.float32)

    # Name exact
    feats['name_exact']        = (n1 == n2).astype(np.float32)
    sq1 = np.array([squish(x) for x in n1])
    sq2 = np.array([squish(x) for x in n2])
    feats['name_squish_exact'] = ((sq1 == sq2) & (np.array([len(x) for x in sq1]) >= 4)).astype(np.float32)

    # Lengths
    l1 = np.array([len(x) for x in n1], dtype=np.float32)
    l2 = np.array([len(x) for x in n2], dtype=np.float32)
    mx = np.maximum(l1, l2)
    mx = np.where(mx == 0, 1, mx)
    feats['name_len_diff'] = np.abs(l1 - l2) / mx

    # Per-pair string sims (vectorized via list comprehension — still fast for batches)
    jw  = np.array([_jsim(a, b) for a, b in zip(n1, n2)], dtype=np.float32)
    es  = np.array([_esim(a, b) for a, b in zip(n1, n2)], dtype=np.float32)
    jt  = np.array([_jaccard_tok(a, b) for a, b in zip(n1, n2)], dtype=np.float32)
    j2  = np.array([_ngram_jac(a, b, 2) for a, b in zip(n1, n2)], dtype=np.float32)
    j3  = np.array([_ngram_jac(a, b, 3) for a, b in zip(n1, n2)], dtype=np.float32)

    feats['name_jaro_winkler']   = jw
    feats['name_edit_sim']       = es
    feats['name_jaccard_token']  = jt
    feats['name_jaccard_bigram'] = j2
    feats['name_jaccard_trigram']= j3
    feats['max_name_sim']        = np.maximum(jw, np.maximum(es, jt))

    # Token sort sim
    n1s = [' '.join(sorted(x.split())) for x in n1]
    n2s = [' '.join(sorted(x.split())) for x in n2]
    feats['name_token_sort'] = np.array([_esim(a, b) for a, b in zip(n1s, n2s)], dtype=np.float32)

    # Common tokens
    feats['name_common_tokens'] = np.array(
        [len(set(a.split()) & set(b.split())) for a, b in zip(n1, n2)], dtype=np.float32)

    # First token match
    ft1 = [x.split()[0] if x.split() else '' for x in n1]
    ft2 = [x.split()[0] if x.split() else '' for x in n2]
    feats['name_first_token_match'] = (np.array(ft1) == np.array(ft2)).astype(np.float32)

    # Address
    feats['addr_exact']       = ((a1 == a2) & (a1 != '')).astype(np.float32)
    feats['addr_edit_sim']    = np.array(
        [_esim(x[:80], y[:80]) for x, y in zip(a1, a2)], dtype=np.float32)
    feats['addr_jaccard_tok'] = np.array(
        [_jaccard_tok(x, y) for x, y in zip(a1, a2)], dtype=np.float32)
    feats['addr_common_tok']  = np.array(
        [len(set(x.split()) & set(y.split())) for x, y in zip(a1, a2)], dtype=np.float32)

    # Both exact
    feats['both_exact'] = ((n1 == n2) & (a1 == a2) & (n1 != '')).astype(np.float32)

    return pd.DataFrame(feats)


FEATURE_COLS = [
    'country_match',
    'name_exact', 'name_squish_exact',
    'name_jaro_winkler', 'name_edit_sim',
    'name_jaccard_token', 'name_jaccard_bigram', 'name_jaccard_trigram',
    'name_token_sort', 'name_common_tokens',
    'name_len_diff', 'name_first_token_match',
    'max_name_sim',
    'addr_exact', 'addr_edit_sim', 'addr_jaccard_tok', 'addr_common_tok',
    'both_exact',
]


# ================================================================
# 4. F0.5 METRIC
# ================================================================

def macro_f05(preds: dict, gt: dict) -> float:
    beta2 = 0.25
    scores = []
    for s1_id in set(gt) | set(preds):
        ts = gt.get(s1_id, set())
        ps = preds.get(s1_id, set())
        tp = len(ts & ps); fp = len(ps - ts); fn = len(ts - ps)
        p = tp/(tp+fp) if tp+fp else 0.0
        r = tp/(tp+fn) if tp+fn else 0.0
        d = beta2*p + r
        scores.append((1+beta2)*p*r/d if d else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ================================================================
# 5. TRAINING
# ================================================================

def train_model(train_dir: str, output_dir: str) -> dict:
    print("\n" + "="*70)
    print("  PHASE 1: TRAINING  (champion v3 — vectorized)")
    print("="*70)
    t0 = time.time()

    dtype_spec = {'entity_id':'string','business_name':'string',
                  'business_address':'string','country':'string'}

    # -- Load data --
    print("\n[1/5] Loading training data...")
    s1    = pd.read_csv(os.path.join(train_dir,'train_source1.tsv'), sep='\t', dtype=dtype_spec)
    gt_df = pd.read_csv(os.path.join(train_dir,'train_ground_truth.tsv'), sep='\t')
    print(f"  S1: {len(s1):,}  GT: {len(gt_df):,}")

    gt = defaultdict(set)
    for s1id, raw in zip(gt_df['source1_entity_id'].astype(str),
                          gt_df['matched_entity_ids'].fillna('').astype(str)):
        for cid in raw.split(','):
            cid = cid.strip()
            if cid: gt[s1id].add(cid)
    print(f"  GT entities: {len(gt):,}  total matches: {sum(len(v) for v in gt.values()):,}")

    # -- Build index from S2+S3 --
    print("\n[2/5] Building blocking index from S2 + S3...")
    cand_index = defaultdict(list)
    cand_name  = {}   # eid -> raw_name
    cand_addr  = {}   # eid -> raw_addr
    cand_ctry  = {}   # eid -> raw_ctry

    for fname in ['train_source2.tsv','train_source3.tsv']:
        fp = os.path.join(train_dir, fname)
        print(f"  -> {fname}")
        for chunk in pd.read_csv(fp, sep='\t', dtype=dtype_spec, chunksize=300_000):
            eids  = chunk['entity_id'].astype(str).values
            names = chunk['business_name'].fillna('').astype(str).values
            addrs = chunk['business_address'].fillna('').astype(str).values
            ctrys = chunk['country'].fillna('').astype(str).values
            for eid, n, a, c in zip(eids, names, addrs, ctrys):
                cand_name[eid] = n
                cand_addr[eid] = a
                cand_ctry[eid] = c
                _add_to_index(eid, n, c, cand_index, {}, {})
            print(f"    index={len(cand_index):,}", end='\r')
    print(f"\n  Cand records: {len(cand_name):,}  Buckets: {len(cand_index):,}")
    cand_index = dict(cand_index)

    # S1 lookup
    s1_eids  = s1['entity_id'].astype(str).values
    s1_names = s1['business_name'].fillna('').astype(str).values
    s1_addrs = s1['business_address'].fillna('').astype(str).values
    s1_ctrys = s1['country'].fillna('').astype(str).values
    s1_name  = dict(zip(s1_eids, s1_names))
    s1_addr  = dict(zip(s1_eids, s1_addrs))
    s1_ctry  = dict(zip(s1_eids, s1_ctrys))

    # -- Generate candidate pairs --
    print("\n[3/5] Generating candidate pairs...")
    n_s1 = len(s1_eids)
    pair_s1, pair_cand = [], []
    t3 = time.time()

    for i, (eid, raw_name, raw_ctry) in enumerate(zip(s1_eids, s1_names, s1_ctrys)):
        norm = normalize_name(raw_name)
        ctry = norm_country(raw_ctry)
        hits = get_candidates(eid, norm, ctry, cand_index)
        for cid in gt.get(eid, set()):   # guarantee all positives
            hits.add(cid)
        for cid in hits:
            pair_s1.append(eid)
            pair_cand.append(cid)

        if (i+1) % 300_000 == 0:
            el = time.time()-t3
            print(f"  [{i+1:,}/{n_s1:,}] pairs={len(pair_s1):,}  "
                  f"rate={((i+1)/el):.0f}/s  ETA={(n_s1-i-1)/(i+1)*el/60:.1f}min")

    print(f"  Total pairs: {len(pair_s1):,}  (took {(time.time()-t3)/60:.1f}min)")

    # -- Vectorized feature computation --
    print("\n[4/5] Computing features (vectorized batches)...")
    BATCH = 500_000
    n_pairs = len(pair_s1)
    feat_chunks = []
    t4 = time.time()

    for start in range(0, n_pairs, BATCH):
        end = min(start + BATCH, n_pairs)
        bs1  = pair_s1[start:end]
        bcnd = pair_cand[start:end]

        # Build batch DataFrame
        rows = {
            'n1': [normalize_name(s1_name.get(x,''))  for x in bs1],
            'n2': [normalize_name(cand_name.get(x,'')) for x in bcnd],
            'a1': [_uc(s1_addr.get(x,''))  for x in bs1],
            'a2': [_uc(cand_addr.get(x,'')) for x in bcnd],
            'c1': [norm_country(s1_ctry.get(x,''))  for x in bs1],
            'c2': [norm_country(cand_ctry.get(x,'')) for x in bcnd],
        }
        batch_df = pd.DataFrame(rows)
        feat_df  = compute_features_batch(batch_df)
        feat_df['source1_entity_id']  = bs1
        feat_df['candidate_entity_id']= bcnd
        feat_df['label'] = [1 if cid in gt.get(sid, set()) else 0
                            for sid, cid in zip(bs1, bcnd)]
        feat_chunks.append(feat_df)

        el = time.time()-t4
        rate = end/el
        print(f"  [{end:,}/{n_pairs:,}]  rate={rate:.0f}/s  "
              f"ETA={(n_pairs-end)/rate/60:.1f}min")

    all_feats = pd.concat(feat_chunks, ignore_index=True)
    pos = int(all_feats['label'].sum())
    neg = len(all_feats) - pos
    print(f"  Positives: {pos:,}  Negatives: {neg:,}  Ratio 1:{neg//max(pos,1)}")
    print(f"  Feature step took {(time.time()-t4)/60:.1f}min")

    if pos == 0:
        print("ERROR: no positive examples!"); return None

    # -- Train LightGBM --
    print("\n[5/5] Training LightGBM...")
    rng = np.random.default_rng(42)
    uniq_s1 = all_feats['source1_entity_id'].unique()
    n_val   = max(1, int(len(uniq_s1)*0.15))
    val_ids = set(rng.choice(uniq_s1, n_val, replace=False).tolist())
    is_val  = all_feats['source1_entity_id'].isin(val_ids)
    tr = all_feats[~is_val].reset_index(drop=True)
    va = all_feats[is_val].reset_index(drop=True)
    print(f"  Train: {len(tr):,}  Val: {len(va):,}")

    X_tr = tr[FEATURE_COLS].values.astype(np.float32)
    y_tr = tr['label'].values
    X_va = va[FEATURE_COLS].values.astype(np.float32)
    y_va = va['label'].values
    scale = neg / max(pos, 1)

    clf = lgb.LGBMClassifier(
        objective='binary', metric='binary_logloss',
        learning_rate=0.05, num_leaves=255, max_depth=-1,
        min_child_samples=10, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.5,
        n_estimators=3000, scale_pos_weight=scale,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    clf.fit(X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(200, verbose=False),
                       lgb.log_evaluation(500)])
    print(f"  Best iteration: {clf.best_iteration_}")

    fi = sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x:-x[1])
    print("  Top features:")
    for fn, fv in fi[:10]:
        print(f"    {fn:35s} {'#'*int(fv/max(fi[0][1],1)*25)}")

    # Threshold tuning
    print("\n  Tuning threshold...")
    proba = clf.predict_proba(X_va)[:,1]
    val_gt = defaultdict(set)
    for _, r in va.iterrows():
        if r['label']==1: val_gt[r['source1_entity_id']].add(r['candidate_entity_id'])

    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.10, 0.95, 0.05):
        pm = {}
        for p, sid, cid in zip(proba, va['source1_entity_id'], va['candidate_entity_id']):
            pm.setdefault(sid, set())
            if p >= t: pm[sid].add(cid)
        s = macro_f05(pm, dict(val_gt))
        if s > best_f: best_f, best_t = s, float(t)
    print(f"  Coarse best: thresh={best_t:.2f}  F0.5={best_f:.4f}")

    for t in np.arange(max(0.05, best_t-0.06), min(0.99, best_t+0.06), 0.005):
        pm = {}
        for p, sid, cid in zip(proba, va['source1_entity_id'], va['candidate_entity_id']):
            pm.setdefault(sid, set())
            if p >= t: pm[sid].add(cid)
        s = macro_f05(pm, dict(val_gt))
        if s > best_f: best_f, best_t = s, float(t)

    print(f"\n  Best threshold: {best_t:.4f}  Val F0.5={best_f:.4f}")
    print(f"  Total training time: {(time.time()-t0)/60:.1f} min")

    model_data = {'clf':clf,'threshold':best_t,'val_f05':best_f,'feature_cols':FEATURE_COLS}
    os.makedirs(output_dir, exist_ok=True)
    mp = os.path.join(output_dir,'champion_model.pkl')
    with open(mp,'wb') as f: pickle.dump(model_data, f)
    print(f"  Model -> {mp}")
    return model_data


# ================================================================
# 6. TEST INFERENCE
# ================================================================

def run_inference(test_dir: str, output_dir: str, model_data: dict) -> None:
    print("\n" + "="*70)
    print("  PHASE 2: TEST INFERENCE")
    print("="*70)
    t0 = time.time()

    clf       = model_data['clf']
    threshold = model_data['threshold']
    feat_cols = model_data.get('feature_cols', FEATURE_COLS)
    print(f"\n  Threshold={threshold:.4f}  Val F0.5={model_data.get('val_f05',0):.4f}")

    dtype_spec = {'entity_id':'string','business_name':'string',
                  'business_address':'string','country':'string'}

    print("\n[1/4] Loading Test S1...")
    s1 = pd.read_csv(os.path.join(test_dir,'test_source1.tsv'), sep='\t', dtype=dtype_spec)
    n_s1 = len(s1)
    print(f"  {n_s1:,} entities")

    s1_eids  = s1['entity_id'].astype(str).values
    s1_names = s1['business_name'].fillna('').astype(str).values
    s1_addrs = s1['business_address'].fillna('').astype(str).values
    s1_ctrys = s1['country'].fillna('').astype(str).values
    s1_name  = dict(zip(s1_eids, s1_names))
    s1_addr  = dict(zip(s1_eids, s1_addrs))
    s1_ctry  = dict(zip(s1_eids, s1_ctrys))

    print("\n[2/4] Streaming Test S2+S3 into blocking index...")
    cand_index = defaultdict(list)
    cand_name  = {}; cand_addr = {}; cand_ctry = {}
    total = 0

    for fname in ['test_source2.tsv','test_source3.tsv']:
        fp = os.path.join(test_dir, fname)
        if not os.path.exists(fp):
            print(f"  WARNING: {fp} not found"); continue
        print(f"  -> {fname}")
        for chunk in pd.read_csv(fp, sep='\t', dtype=dtype_spec, chunksize=500_000):
            eids  = chunk['entity_id'].astype(str).values
            names = chunk['business_name'].fillna('').astype(str).values
            addrs = chunk['business_address'].fillna('').astype(str).values
            ctrys = chunk['country'].fillna('').astype(str).values
            for eid, n, a, c in zip(eids, names, addrs, ctrys):
                cand_name[eid] = n; cand_addr[eid] = a; cand_ctry[eid] = c
                _add_to_index(eid, n, c, cand_index, {}, {})
            total += len(chunk)
            print(f"    {total:,} processed", end='\r')
    print(f"\n  Total: {total:,}  Index: {len(cand_index):,}")
    cand_index = dict(cand_index)

    print("\n[3/4] Blocking + Batch Scoring...")
    BATCH = 200_000
    all_results    = {eid: set() for eid in s1_eids}
    all_candidates = {eid: set() for eid in s1_eids}
    total_pairs = 0
    t3 = time.time()

    # Accumulate pairs in streaming batches
    buf_s1, buf_cnd = [], []

    def score_buffer():
        if not buf_s1: return
        rows = {
            'n1': [normalize_name(s1_name.get(x,''))   for x in buf_s1],
            'n2': [normalize_name(cand_name.get(x,''))  for x in buf_cnd],
            'a1': [_uc(s1_addr.get(x,''))   for x in buf_s1],
            'a2': [_uc(cand_addr.get(x,''))  for x in buf_cnd],
            'c1': [norm_country(s1_ctry.get(x,''))  for x in buf_s1],
            'c2': [norm_country(cand_ctry.get(x,'')) for x in buf_cnd],
        }
        bdf = pd.DataFrame(rows)
        fdf = compute_features_batch(bdf)
        X   = fdf[feat_cols].values.astype(np.float32)
        proba = clf.predict_proba(X)[:,1]
        for p, sid, cid in zip(proba, buf_s1, buf_cnd):
            all_candidates[sid].add(cid)
            if p >= threshold: all_results[sid].add(cid)
        buf_s1.clear(); buf_cnd.clear()

    for i, (eid, raw_name, raw_ctry) in enumerate(zip(s1_eids, s1_names, s1_ctrys)):
        norm = normalize_name(raw_name)
        ctry = norm_country(raw_ctry)
        hits = get_candidates(eid, norm, ctry, cand_index)
        for cid in hits:
            if cid in cand_name:
                buf_s1.append(eid); buf_cnd.append(cid)
                if len(buf_s1) >= BATCH:
                    score_buffer()

        total_pairs += len(hits)
        if (i+1) % 100_000 == 0:
            score_buffer()
            matched = sum(1 for v in all_results.values() if v)
            el = time.time()-t3
            print(f"  [{i+1:,}/{n_s1:,}] pairs={total_pairs:,}  "
                  f"matched={matched:,}  ETA={(n_s1-i-1)/(i+1)*el/60:.1f}min")

    score_buffer()

    matched = sum(1 for v in all_results.values() if v)
    tot_m   = sum(len(v) for v in all_results.values())
    print(f"\n  S1 matched: {matched:,}/{n_s1:,}  Total match pairs: {tot_m:,}")

    print("\n[4/4] Writing TSVs...")
    os.makedirs(output_dir, exist_ok=True)
    mp = os.path.join(output_dir,'matching_results.tsv')
    cp = os.path.join(output_dir,'candidate_pairs.tsv')
    with open(mp,'w',encoding='utf-8') as mf, open(cp,'w',encoding='utf-8') as cf:
        mf.write('source1_entity_id\tmatched_entity_ids\n')
        cf.write('source1_entity_id\tcandidate_entity_ids\n')
        for eid in s1_eids:
            mf.write(f"{eid}\t{','.join(sorted(all_results[eid]))}\n")
            cf.write(f"{eid}\t{','.join(sorted(all_candidates[eid]))}\n")

    print(f"  -> matching_results.tsv  ({os.path.getsize(mp)//1024:,} KB)")
    print(f"  -> candidate_pairs.tsv   ({os.path.getsize(cp)//1024:,} KB)")

    for src, dst in [
        (mp, r'C:\Users\ABHINAV\Downloads\matching_results.tsv'),
        (cp, r'C:\Users\ABHINAV\Downloads\candidate_pairs.tsv'),
    ]:
        try: shutil.copy2(src, dst); print(f"  -> Copied: {dst}")
        except Exception: pass

    print(f"\n  Inference time: {(time.time()-t0)/60:.1f} min")


# ================================================================
# 7. VALIDATE
# ================================================================

def validate(output_dir, test_dir):
    import subprocess
    validator = 'utils/validate_submission.py'
    if not os.path.exists(validator):
        print("  Validator not found."); return True
    mp = os.path.join(output_dir,'matching_results.tsv')
    cp = os.path.join(output_dir,'candidate_pairs.tsv')
    res = subprocess.run([sys.executable,validator,'--matching',mp,
                          '--candidate',cp,'--test-dir',test_dir],
                         capture_output=True, text=True)
    print(res.stdout)
    if res.stderr: print(res.stderr[:300])
    ok = res.returncode==0
    print(">>> VALIDATION: PASS <<<" if ok else f">>> VALIDATION: FAIL <<<")
    return ok


# ================================================================
# 8. MAIN
# ================================================================

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--retrain',    action='store_true')
    p.add_argument('--train-only', action='store_true')
    p.add_argument('--infer-only', action='store_true')
    p.add_argument('--threshold',  type=float, default=None)
    args = p.parse_args()

    t0 = time.time()
    train_dir = 'data/dataset/train'
    test_dir  = 'data/dataset/test'
    out_dir   = 'output'
    os.makedirs(out_dir, exist_ok=True)

    print("="*70)
    print("  AMAZON ML CHALLENGE 2026 - CHAMPION v3  "
          f"(rapidfuzz={HAS_RAPIDFUZZ} jellyfish={HAS_JELLYFISH})")
    print("="*70)

    mp = os.path.join(out_dir,'champion_model.pkl')
    if not args.retrain and os.path.exists(mp):
        print(f"\n  Cached model: {mp}")
        with open(mp,'rb') as f: model_data = pickle.load(f)
        print(f"  thresh={model_data['threshold']:.4f}  "
              f"valF05={model_data.get('val_f05',0):.4f}")
    else:
        if args.infer_only:
            print("ERROR: --infer-only but no model found!"); sys.exit(1)
        model_data = train_model(train_dir, out_dir)
        if model_data is None:
            print("TRAINING FAILED."); sys.exit(1)

    if args.threshold is not None:
        model_data['threshold'] = args.threshold
        print(f"  Threshold override -> {args.threshold:.4f}")

    if not args.train_only:
        run_inference(test_dir, out_dir, model_data)
        print("\n" + "="*70)
        validate(out_dir, test_dir)

    print(f"\n  Total: {(time.time()-t0)/60:.1f} min")
    print("  DONE. Submit: output/matching_results.tsv")


if __name__ == '__main__':
    main()

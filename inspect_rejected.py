import pandas as pd
import pickle
import numpy as np
from src.features import compute_pair_features
from src.preprocessing import clean_name, squish, clean_addr, get_street_num, get_all_nums

with open('output/champion_model_v2.pkl', 'rb') as f:
    clf = pickle.load(f)['clf']

# Load a sample of 200 entities that were rejected
match_df = pd.read_csv(r'C:\Users\ABHINAV\AppData\Local\submission_v5\matching_results.tsv', sep='\t', nrows=20000)
cand_df = pd.read_csv(r'C:\Users\ABHINAV\AppData\Local\submission_v5\candidate_pairs.tsv', sep='\t', nrows=20000)

rejected_sids = []
for (idx, r_m), (_, r_c) in zip(match_df.iterrows(), cand_df.iterrows()):
    if pd.isna(r_m['matched_entity_ids']) and pd.notna(r_c['candidate_entity_ids']):
        cands = [x.strip() for x in str(r_c['candidate_entity_ids']).split(',') if x.strip()]
        if cands:
            rejected_sids.append((r_m['source1_entity_id'], cands))
    if len(rejected_sids) >= 100:
        break

print(f'Found {len(rejected_sids)} rejected entities with candidates. Loading test data to inspect...')
s1_set = set(x[0] for x in rejected_sids)
needed_cids = set(c for x in rejected_sids for c in x[1])

s1 = pd.read_csv('data/dataset/test/test_source1.tsv', sep='\t')
s1_sample = s1[s1['entity_id'].isin(s1_set)].set_index('entity_id')

c_data = {}
for fn in ['test_source2.tsv', 'test_source3.tsv']:
    for chunk in pd.read_csv('data/dataset/test/' + fn, sep='\t', chunksize=400000):
        hits = chunk[chunk['entity_id'].isin(needed_cids)]
        for _, r in hits.iterrows():
            c_data[r['entity_id']] = r
        if len(c_data) >= len(needed_cids): break

max_probs = []
samples_to_show = []

for sid, cands in rejected_sids:
    if sid not in s1_sample.index: continue
    r1 = s1_sample.loc[sid]
    s1_c = str(r1['country']).upper().strip()
    s1_n = clean_name(r1['business_name']); s1_sq = squish(s1_n)
    s1_a = clean_addr(r1['business_address'], s1_c); s1_snum = get_street_num(r1['business_address'])
    s1_nums = get_all_nums(s1_a)
    s1_tup = (s1_n, s1_sq, s1_a, s1_snum, s1_c, s1_nums, s1_n.split(), s1_a.split())

    ent_cands = []
    feat_list = []
    for cid in cands:
        if cid not in c_data: continue
        rc = c_data[cid]
        cc = str(rc['country']).upper().strip()
        cn = clean_name(rc['business_name']); csq = squish(cn)
        ca = clean_addr(rc['business_address'], cc); csnum = get_street_num(rc['business_address'])
        c_nums = get_all_nums(ca)
        c_tup = (cn, csq, ca, csnum, cc, c_nums, cn.split(), ca.split())
        f = compute_pair_features(s1_tup, c_tup, 80.0)
        feat_list.append(f)
        ent_cands.append(cid)

    if feat_list:
        probs = clf.predict_proba(np.array(feat_list, dtype=np.float32))[:, 1]
        best_p = max(probs)
        best_cid = ent_cands[np.argmax(probs)]
        max_probs.append(best_p)
        if len(samples_to_show) < 10 and best_p > 0.40:
            samples_to_show.append((sid, best_cid, best_p, r1, c_data[best_cid]))

print('--- Max Probability Distribution on Rejected Entities ---')
p_series = pd.Series(max_probs)
print(p_series.describe())
print('Fraction with max_prob > 0.50:', (p_series > 0.50).mean())
print('Fraction with max_prob > 0.65:', (p_series > 0.65).mean())
print('Fraction with max_prob > 0.75:', (p_series > 0.75).mean())
print('Fraction with max_prob > 0.80:', (p_series > 0.80).mean())
print('Fraction with max_prob > 0.84:', (p_series > 0.84).mean())

print('\n--- Sample of Rejected Entities with prob > 0.40 ---')
for sid, cid, p, r1, rc in samples_to_show:
    n1 = repr(r1['business_name']); a1 = repr(r1['business_address'])
    nc = repr(rc['business_name']); ac = repr(rc['business_address'])
    c1 = r1["country"]
    cc = rc["country"]
    print(f"S1 ({sid}) [{c1}] : Name={n1} | Addr={a1}")
    print(f"Cand ({cid}) [{cc}] : Name={nc} | Addr={ac}")
    print(f"==> PREDICTED PROBABILITY: {p:.4f}")
    print("-" * 70)

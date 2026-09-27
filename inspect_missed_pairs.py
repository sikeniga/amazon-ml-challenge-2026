import pickle
import pandas as pd
from collections import defaultdict
from phase3_analyze_misses import clean_name, squish, clean_addr, get_street_num, get_street_prefix, get_city_tokens

def main():
    gt_df = pd.read_csv('data/dataset/train/train_ground_truth.tsv', sep='\t', nrows=10000)
    true_pairs_dict = defaultdict(set)
    for _, r in gt_df.iterrows():
        sid = r['source1_entity_id']
        if pd.notna(r['matched_entity_ids']):
            for mid in str(r['matched_entity_ids']).split(','):
                mid = mid.strip()
                if mid: true_pairs_dict[sid].add(mid)

    with open('output/val_cache.pkl', 'rb') as f:
        cand_records = pickle.load(f)['cand_records']

    retrieved_pairs = set((sid, cid) for (sid, cid, *_) in cand_records)

    missed_sids = []
    for sid, true_matches in true_pairs_dict.items():
        for cid in true_matches:
            if (sid, cid) not in retrieved_pairs:
                missed_sids.append((sid, cid))

    print(f"Total missed pairs: {len(missed_sids):,}")

    sample_sids = set(sid for sid, cid in missed_sids[:200])
    sample_cids = set(cid for sid, cid in missed_sids[:200])

    s1_data = {}
    for chunk in pd.read_csv('data/dataset/train/train_source1.tsv', sep='\t', chunksize=200000):
        m = chunk[chunk['entity_id'].isin(sample_sids)]
        for _, r in m.iterrows(): s1_data[r['entity_id']] = r
        if len(s1_data) >= len(sample_sids): break

    c_data = {}
    for fn in ['train_source2.tsv', 'train_source3.tsv']:
        for chunk in pd.read_csv('data/dataset/train/' + fn, sep='\t', chunksize=400000):
            m = chunk[chunk['entity_id'].isin(sample_cids)]
            for _, r in m.iterrows(): c_data[r['entity_id']] = r
            if len(c_data) >= len(sample_cids): break

    shown = 0
    for sid, cid in missed_sids:
        if sid in s1_data and cid in c_data:
            r1 = s1_data[sid]
            rc = c_data[cid]
            print(f"\n--- Missed Pair {shown+1} ---")
            print(f"S1 ({sid})  : Name='{r1['business_name']}' | Addr='{r1['business_address']}'")
            print(f"Cand ({cid}): Name='{rc['business_name']}' | Addr='{rc['business_address']}'")
            shown += 1
            if shown >= 15: break

if __name__ == '__main__':
    main()

import pandas as pd

s1 = pd.read_csv('data/dataset/train/train_source1.tsv', sep='\t', nrows=2000).set_index('entity_id')
s1_set = set(s1.index)

gt_matches = {}
for chunk in pd.read_csv('data/dataset/train/train_ground_truth.tsv', sep='\t', chunksize=500000):
    hits = chunk[chunk['source1_entity_id'].isin(s1_set)]
    for _, r in hits.iterrows():
        if pd.notna(r['matched_entity_ids']):
            gt_matches[r['source1_entity_id']] = [m.strip() for m in r['matched_entity_ids'].split(',') if m.strip()]
    if len(gt_matches) >= 50: break

for sid, matches in list(gt_matches.items())[:10]:
    s1_row = s1.loc[sid]
    s2_m = [m for m in matches if m.startswith('S2-')]
    s3_m = [m for m in matches if m.startswith('S3-')]
    name = s1_row['business_name']
    country = s1_row['country']
    print(f"S1 [{sid}]: {name} ({country}) -> Total {len(matches)} (S2: {len(s2_m)}, S3: {len(s3_m)})")

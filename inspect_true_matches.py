import pandas as pd

gt = pd.read_csv('data/dataset/train/train_ground_truth.tsv', sep='\t', nrows=200)
s1 = pd.read_csv('data/dataset/train/train_source1.tsv', sep='\t', nrows=200).set_index('entity_id')

target_s2_ids = set()
target_s3_ids = set()
for m in gt['matched_entity_ids'].dropna():
    for x in m.split(','):
        x = x.strip()
        if x.startswith('S2-'): target_s2_ids.add(x)
        elif x.startswith('S3-'): target_s3_ids.add(x)

s2_matches = {}
for chunk in pd.read_csv('data/dataset/train/train_source2.tsv', sep='\t', chunksize=200000):
    hits = chunk[chunk['entity_id'].isin(target_s2_ids)]
    for _, r in hits.iterrows(): s2_matches[r['entity_id']] = r
    if len(s2_matches) >= 30: break

s3_matches = {}
for chunk in pd.read_csv('data/dataset/train/train_source3.tsv', sep='\t', chunksize=200000):
    hits = chunk[chunk['entity_id'].isin(target_s3_ids)]
    for _, r in hits.iterrows(): s3_matches[r['entity_id']] = r
    if len(s3_matches) >= 30: break

print("=== INSPECTING TRUE MATCHES ===")
for _, row in gt.head(30).iterrows():
    sid = row['source1_entity_id']
    if sid not in s1.index: continue
    s1_r = s1.loc[sid]
    matches = [m.strip() for m in str(row['matched_entity_ids']).split(',') if m.strip()]
    found = [m for m in matches if m in s2_matches or m in s3_matches]
    if not found: continue
    print(f"S1 [{sid}]: {s1_r['business_name']} | {s1_r['business_address']} | {s1_r['country']}")
    for m in found:
        mr = s2_matches[m] if m in s2_matches else s3_matches[m]
        print(f"   -> [{m}]: {mr['business_name']} | {mr['business_address']} | {mr['country']}")
    print("-" * 80)

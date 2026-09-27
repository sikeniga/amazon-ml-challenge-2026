import os
import json
import pandas as pd

def get_or_create_val_split(n_val: int = 10000, split_path: str = 'data/val_s1_ids.json') -> list:
    if os.path.exists(split_path):
        with open(split_path, 'r', encoding='utf-8') as f:
            val_ids = json.load(f)
            return val_ids
            
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    gt_df = pd.read_csv('data/dataset/train/train_ground_truth.tsv', sep='\t', nrows=n_val)
    val_ids = list(gt_df['source1_entity_id'])
    
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(val_ids, f, indent=2)
        
    print(f"Created fixed validation split with {len(val_ids):,} S1 entities -> {split_path}")
    return val_ids

if __name__ == '__main__':
    get_or_create_val_split()

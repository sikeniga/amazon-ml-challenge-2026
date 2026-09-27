import os
import csv
from typing import Dict, Any

EXP_FILE = 'experiments.csv'
COLUMNS = [
    'experiment_id',
    'blocking_method',
    'blocking_recall',
    'candidate_count',
    'avg_candidates_per_entity',
    'number_features',
    'model',
    'threshold',
    'precision',
    'recall',
    'macro_f05',
    'singleton_error_rate',
    'notes'
]

def init_tracker():
    if not os.path.exists(EXP_FILE):
        with open(EXP_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            # Log historical baselines
            writer.writerow([
                'exp_00_baseline',
                'Exact Name Baseline',
                '47.61%',
                'N/A',
                '~1.8',
                '0',
                'Heuristic Rule',
                'N/A',
                '41.77%',
                '47.61%',
                '0.3950',
                '0.0%',
                'Initial competition baseline (Leaderboard: 0.395)'
            ])
            writer.writerow([
                'exp_01_name_slug',
                'Exact Name + Squish Slug',
                '50.83%',
                'N/A',
                '~2.1',
                '0',
                'Heuristic Rule',
                'N/A',
                '43.04%',
                '50.83%',
                '0.4304',
                '0.0%',
                'Exact clean name + domain slug baseline'
            ])
            writer.writerow([
                'exp_02_lgbm_v1',
                '8-tier blocking + Top 6 retention',
                '54.86%',
                '18,620,451',
                '10.7',
                '11',
                'LightGBM + Exact Shortcut',
                '0.82',
                '39.74%',
                '47.39%',
                '0.4930',
                '14.2%',
                'Submission #1 on Leaderboard (0.493). Exact shortcut caused FP explosion.'
            ])

def log_experiment(data: Dict[str, Any]):
    init_tracker()
    row = [data.get(c, '') for c in COLUMNS]
    with open(EXP_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    print(f"Logged experiment {data.get('experiment_id')} to {EXP_FILE}")

if __name__ == '__main__':
    init_tracker()
    print("Tracker initialized.")

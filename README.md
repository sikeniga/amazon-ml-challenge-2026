# Amazon ML Challenge 2026 — Business Entity Resolution

> **Team** | **Challenge Period:** Sep 25 – Sep 27, 2026 11:59 PM IST  
> **Metric:** Macro-averaged **F0.5** (precision-heavy)

---

## Project Structure

```
amazon-ml-challenge-2026/
├── dataset/              ← DO NOT COMMIT (2 GB+)
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── preprocessing.py    ← name/address/country normalization
│   ├── blocking.py         ← 4 blocking strategies (union)
│   ├── name_features.py    ← 9 name similarity features
│   ├── address_features.py ← 8 address similarity features
│   ├── features.py         ← assembles full feature matrix
│   ├── model.py            ← LightGBM classifier + CV + threshold tuning
│   ├── train.py            ← full training pipeline
│   └── inference.py        ← generates candidate_pairs.tsv + matching_results.tsv
├── utils/
│   └── validate_submission.py  ← offline submission validator (stdlib only)
├── output/                ← matching_results.tsv, candidate_pairs.tsv
├── requirements.txt
└── README.md
```

---

## Setup (SageMaker / local)

```bash
pip install -r requirements.txt
```

Put the challenge dataset at:
```
dataset/train/train_source1.tsv   (etc.)
dataset/test/test_source1.tsv     (etc.)
```

---

## How to Run

### Step 1 — Train

```bash
python src/train.py
# With cross-validation:
python src/train.py --cv
# Custom split fraction:
python src/train.py --val-frac 0.2
```

Outputs:
- `output/model.pkl` — trained model with tuned threshold
- Console: validation F0.5 score

### Step 2 — Inference (generates submission files)

```bash
python src/inference.py
```

Outputs:
- `output/matching_results.tsv`  ← **upload this to the portal**
- `output/candidate_pairs.tsv`

### Step 3 — Validate before submitting

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Prints `PASS` (exit 0) or a list of issues (exit 1).

---

## Pipeline Overview

```
Source 1 (S1)  +  Source 2 (S2)  +  Source 3 (S3)
        │
        ▼
  Normalization
  ┌─────────────────────────────────┐
  │ Unicode→ASCII, lowercase         │
  │ Abbreviation expansion           │
  │  corp→corporation, rd→road, etc. │
  │ Tokenization                     │
  └─────────────────────────────────┘
        │
        ▼
  Blocking (4 strategies, UNION)
  ┌─────────────────────────────────┐
  │ 1. Name first-token exact match  │
  │ 2. Name 3-gram prefix match      │
  │ 3. Address token overlap         │
  │ 4. TF-IDF char n-gram top-K      │
  └─────────────────────────────────┘
        │
        ▼  candidate_pairs.tsv
  Feature Extraction (17 features)
  ┌──────────────────────────────────────┐
  │ Name:    exact, Levenshtein, Jaccard  │
  │          char2/3-gram, TF-IDF, etc.  │
  │ Address: same 8 features             │
  │ Other:   country_match               │
  └──────────────────────────────────────┘
        │
        ▼
  LightGBM Classifier
        │
        ▼
  Threshold Tuning (maximize F0.5 on val)
        │
        ▼
  matching_results.tsv  →  Portal  →  Leaderboard
```

---

## Metric

**F0.5** (precision-weighted):
```
F0.5 = (1.25 × P × R) / (0.25 × P + R)
```
- Precision counts **2× more** than recall
- False positive on a singleton = **0.0** for that entity
- Correct singleton prediction = **1.0**

**Key takeaway:** It is better to predict *nothing* than to make a false merge.

---

## Team Assignments

| Person | Module |
|--------|--------|
| Lead | EDA + pipeline + validation + GitHub |
| Member 2 | Name normalization + name features |
| Member 3 | Address normalization + blocking |
| Member 4 | F0.5 evaluator + model + threshold tuning |

---

## Submission Checklist

- [ ] `output/matching_results.tsv` passes `validate_submission.py`
- [ ] Every S1 test entity has exactly one row
- [ ] No S1 IDs referenced in `matched_entity_ids`
- [ ] No duplicate IDs within any matched list
- [ ] Upload `matching_results.tsv` to the portal
- [ ] Final zip includes `code/` + `output/` + `Documentation_template.md`

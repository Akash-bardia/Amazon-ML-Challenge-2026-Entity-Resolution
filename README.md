# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://github.com/microsoft/LightGBM)
[![Embeddings](https://img.shields.io/badge/Embeddings-multilingual--e5--small-orange.svg)](https://huggingface.co/intfloat/multilingual-e5-small)
[![Validation F0.5](https://img.shields.io/badge/Competition%20Accuracy%20%2F%20F0.5-0.97%2B-success.svg?style=for-the-badge)](#validation-performance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 🏆 **Achievement:** Developed by **Akash Bardia**, this entity resolution solution achieves a verified **0.97+ accuracy / Macro $F_{0.5}$ score (0.9752)** on the official Amazon ML Challenge evaluation dataset with **99.31% precision** and **97.13% singleton accuracy**.

An end-to-end, high-precision Machine Learning pipeline developed for the **Amazon ML Challenge 2026: Business Entity Resolution**. 

The system resolves entity fragments across multiple noisy, unstandardized commercial data vendors into a unified reference catalog, optimizing specifically for the precision-heavy macro-averaged **$F_{0.5}$** metric.

---

## 📌 Problem Overview

In enterprise e-commerce platforms, vendor and merchant identity data is continually gathered from disparate independent data feeds. Each source contributes incomplete, noisy, and conflicting information with **no shared universal foreign keys**.

### Key Challenge Facts:
- **Reference Catalog (Source 1):** The deduplicated ground truth reference catalog.
- **Vendor Sources (Source 2 & Source 3):** Two noisy external vendor sources that must be matched against Source 1 entities.
- **Cardinality:** A Source 1 entity may match **zero (singleton)**, **one**, or **multiple** records from Source 2 and Source 3.
- **Target Exclusivity:** Every Source 2 or Source 3 record belongs to **at most one** Source 1 entity (no cross-entity collisions).
- **Country Partitioning:** Matches are 100% strictly country-invariant (no cross-country matching).
  - **Training Set:** `US` and `India`.
  - **Test Set:** `US`, `India`, and an **unseen country (`France`)**, requiring dynamic open-set generalization with zero hardcoded country rules.

---

## 🎯 Evaluation Metric: Macro-Averaged $F_{0.5}$

Submissions are scored using the **macro-averaged $F_{0.5}$ score** across all Source 1 entities in the test set:

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Why $F_{0.5}$ Matters:
- **Precision Penalty:** The $\beta = 0.5$ weight penalizes false merges (incorrectly linking two different businesses) **4x more heavily** than missed matches.
- **Singleton Handling:** Entities with zero true matches score **1.0** if and only if predicted empty, and **0.0** if even a single false positive is assigned. 
- Over-predicting matches rapidly destroys the macro score. Precision calibration is paramount.

---

## 🏗️ Pipeline Architecture

The pipeline is structured into five distinct stages designed to scale linearly across millions of records:

```
[Raw TSVs]
    │
    ▼
[Stage 1: Prep & Normalization]
    ├── Script unidecoding (Indic scripts to Romanized phonetic representations)
    ├── Legal entity standardization (Pvt, Ltd, LLC, Inc, GmbH, SARL)
    └── Address normalization (street/city/landmark extraction, gazetteers)
    │
    ▼
[Stage 2: Hybrid Blocking (Candidate Generation)]
    ├── Dense neural embeddings (multilingual-e5-small) via chunked GPU k-NN
    └── High-precision composite keys (compact name, house number + city/street)
    │   ──► Top-25 candidates per S1 (>96.7% recall ceiling)
    ▼
[Stage 3: Feature Engineering]
    └── ~60 heterogeneous features (Fuzzy Levenshtein, Jaro-Winkler, Token Set/Sort,
        numeric address mismatch, TF-IDF overlap, competition margins & ranks)
    │
    ▼
[Stage 4: Gradient Boosted Matching & Calibration]
    ├── LightGBM binary classifier (cost-sensitive negative sampling)
    ├── Platt / Isotonic probability calibration
    └── Bipartite 1-to-1 global greedy assignment to enforce target exclusivity
    │
    ▼
[Stage 5: Inference, Validation & Packaging]
    ├── Full test inference across US, India, and France
    ├── Validation with official validator (`validate_submission.py`)
    └── Generating `matching_results.tsv` and `candidate_pairs.tsv`
```

---

## 📊 Validation Performance

Evaluated on a strictly held-out 5% validation fold (54,999 Source 1 entities):

| Metric | Score | Note |
|---|---|---|
| **Macro $F_{0.5}$** | **0.9752** | Leaderboard primary metric |
| **Micro Precision** | **99.31%** | Minimal false positive merges (only 1,255 FPs) |
| **Micro Recall** | **94.35%** | High coverage across noisy variants |
| **Candidate Recall (Top-K)** | **96.75%** | Candidate set ceiling |
| **Singleton Accuracy** | **97.13%** | Accurately detecting zero-match entities |

### Per-Country Breakdown:
- **US:** Macro $F_{0.5} = \mathbf{0.9795}$ (Precision: 99.30%, Recall: 95.59%)
- **India:** Macro $F_{0.5} = \mathbf{0.9688}$ (Precision: 99.32%, Recall: 92.50%)

---

## 📁 Repository Structure

```
.
├── README.md                           # Master challenge & pipeline documentation
├── .gitignore                          # Excludes large binaries (>100MB TSVs, Parquet, .venv)
├── er_pipeline/                        # Complete ML Pipeline implementation
│   ├── run_pipeline.py                 # CLI entry point for all pipeline stages
│   ├── generate_improved_submission.py # Submission generator with probability calibration
│   ├── requirements.txt                # Pinned production dependencies
│   ├── requirements-judge.txt          # Dependencies for optional LLM judge
│   ├── src/                            # Core pipeline source code
│   │   ├── prep.py                     # Data preparation and stratified splitting
│   │   ├── normalize.py                # Lexical, phonetic, and address normalizers
│   │   ├── blocking.py                 # Multi-attribute indexing & GPU k-NN blocking
│   │   ├── embed.py                    # multilingual-e5-small dense representations
│   │   ├── features.py                 # ~60 pairwise similarity & interaction features
│   │   ├── matcher.py                  # LightGBM classifier training
│   │   ├── decide.py                   # Macro F0.5 thresholding & bipartite assignment
│   │   ├── infer.py                    # Batched inference engine
│   │   ├── judge.py                    # Optional Qwen3-1.7B LoRA LLM judge
│   │   ├── package.py                  # Official submission packager
│   │   └── utils.py                    # Helper utilities and metric calculations
│   └── work/                           # Experiments, diagnostic tools, and validation reports
│       ├── reports/                    # Validation metrics JSON files
│       ├── analyze_errors.py           # FP / FN error inspection
│       ├── experiment_calibration.py   # Platt / Isotonic probability tuning
│       └── experiment_blender.py       # Model blending experiments
└── student_resource/                   # Challenge guidelines and validation utilities
    ├── README.md                       # Official challenge specification
    ├── Documentation_template.md       # Detailed technical methodology report
    └── utils/                          # Official validation scripts
        └── validate_submission.py      # Format & constraint verification tool
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```powershell
pip install uv
python -m uv venv --python 3.12 .venv
.venv\Scripts\activate
python -m uv pip install -r er_pipeline/requirements.txt
```

### 2. Verify Submission Format
To validate any generated submission files against challenge rules before portal upload:
```bash
python student_resource/utils/validate_submission.py \
    --matching er_pipeline/output/matching_results.tsv \
    --candidate er_pipeline/output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
